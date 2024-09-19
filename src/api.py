import sys
import os
from typing import List, Any, Optional
import time
from functools import partial
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import json
import io

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

import src.client_utils as client

# Parse environment variables
#
load_dotenv()
model_name    = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION", "")
batch_size    = int(os.getenv("BATCH_SIZE", 1))
#
url           = os.getenv("TRITON_URL", "localhost:7000")
protocol      = os.getenv("PROTOCOL", "HTTP")
verbose       = os.getenv("VERBOSE", "False").lower() in ("true", "1", "t")
async_set     = os.getenv("ASYNC_SET", "False").lower() in ("true", "1", "t")
artifacts     = os.getenv('ARTIFACTS', 'artifacts')
os.makedirs(artifacts, exist_ok=True)

try:
    if protocol.lower() == "grpc":
        # Create gRPC client for communicating with the server
        triton_client = grpcclient.InferenceServerClient(
            url=url, verbose=verbose
        )
    else:
        # Specify large enough concurrency to handle the number of requests.
        concurrency = 20 if async_set else 1
        triton_client = httpclient.InferenceServerClient(
            url=url, verbose=verbose, concurrency=concurrency,
            connection_timeout=10000, network_timeout=10000
        )
except Exception as e:
    print("client creation failed: " + str(e))
    sys.exit(1)

try:
    model_metadata = triton_client.get_model_metadata(
        model_name=model_name, model_version=model_version
    )
    model_config = triton_client.get_model_config(
        model_name=model_name, model_version=model_version
    )
except InferenceServerException as e:
    print("failed to retrieve model metadata: " + str(e))
    sys.exit(1)

if protocol.lower() == "grpc":
    model_config = model_config.config
else:
    model_metadata, model_config = client.convert_http_metadata_config(
        model_metadata, model_config
    )

# parsing information of model
max_batch_size, input_names, output_names, formats, dtypes = client.parse_model(
    model_metadata, model_config
)

supports_batching = max_batch_size > 0
if not supports_batching and batch_size != 1:
    print("ERROR: This model doesn't support batching.")
    sys.exit(1)


class InferRequest(BaseModel):
    prompts: List[str]
    negative_prompts: List[str]
    pose_image: Any
    scheduler: str
    steps: int
    guidance_scale: float
    cnet_conditional_scale: float
    seed: int

############
# FastAPI
############


app = FastAPI()
app.mount("/artifacts", StaticFiles(directory=artifacts), name="data")

@app.get("/")   
def root():
    return {"Hello": "World"}

@app.post("/inference")
async def inference(
    prompt: str,
    negative_prompt: str,
    pose_image: UploadFile,
    scheduler: Optional[str] = None,
    steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    cnet_conditional_scale: Optional[float] = None,
    seed: Optional[int] = None,
):
    
    image_content = await pose_image.read()
    image_content = Image.open(io.BytesIO(image_content)).convert('RGB')
    image_R, image_G, image_B = image_content.split()
    image_content_R = image_R.resize((512, 512))
    image_content_G = image_G.resize((512, 512))
    image_content_B = image_B.resize((512, 512))
    image_content = Image.merge("RGB", (image_content_R, image_content_G, image_content_B))
    # image_content.resize((512, 512))
    image_content.save(artifacts + '/pose_image.jpg')

    image_content = np.array(image_content, dtype=np.uint8)
    # low_threshold = 100
    # high_threshold = 200

    # image = cv2.Canny(image, low_threshold, high_threshold)
    # image = image[:, :, None]
    # image = np.concatenate([image, image, image], axis=2)
    canny_image = np.array(image_content)
    print("canny_image.shape: ", canny_image.shape)
    assert canny_image.shape == (512, 512, 3)

    # create default inputs
    """
    prompt = "Monalisa in the red sky"
    negative_prompt = "bad quality"
    """
    scheduler = "PNDMScheduler" if scheduler is None else scheduler
    steps = 20 if steps is None else steps
    guidance_scale = 1.0 if guidance_scale is None else guidance_scale
    cnet_conditional_scale = 1.0 if cnet_conditional_scale is None else cnet_conditional_scale
    seed = -1 if seed is None else seed

    # create inferReqeust
    inferRequest = InferRequest(
        prompts=[prompt],
        negative_prompts=[negative_prompt],
        pose_image=canny_image,
        scheduler=scheduler,
        steps=steps,
        guidance_scale=guidance_scale,
        cnet_conditional_scale=cnet_conditional_scale,
        seed=seed,
    )
    # return the image paths
    response = await diffusers(inferRequest)
    # image_paths = json.loads(response.body)
    return response



@app.post("/diffusers")
async def diffusers(inferRequest: InferRequest) -> JSONResponse:

    # model input params
    sd_inputs = [
        np.array(inferRequest.prompts, dtype=np.object_),
        np.array(inferRequest.negative_prompts, dtype=np.object_),
        np.array(inferRequest.pose_image, dtype=np.float32), #! need to read from file
        np.array([inferRequest.scheduler], dtype=np.object_),
        np.array([inferRequest.steps], dtype=np.int32),
        np.array([inferRequest.guidance_scale], dtype=np.float32),
        np.array([inferRequest.cnet_conditional_scale], dtype=np.float32),
        np.array([inferRequest.seed], dtype=np.int64),
    ]


    # Generate the request
    inputs, outputs = requestGenerator(
        sd_inputs, input_names, output_names, dtypes
    )
    # Perform inference
    try:
        start_time = time.time()

        if protocol.lower() == "grpc":
            user_data = client.UserData()
            output_images = triton_client.async_infer(
                model_name,
                inputs,
                partial(client.completion_callback, user_data),
                model_version=model_version,
                outputs=outputs,
            )
        else:
            async_request = triton_client.async_infer(
                model_name,
                inputs,
                model_version=model_version,
                outputs=outputs,
            )
    except InferenceServerException as e:
        return {"Error": "Inference failed with error: " + str(e)}

    # Collect results from the ongoing async requests
    if protocol.lower() == "grpc":
        (output_images, error) = user_data._completed_requests.get()
        if error is not None:
            return {"Error": "Inference failed with error: " + str(error)}
    else:
        # HTTP
        output_images = async_request.get_result()

    # Process the results    
    end_time = time.time()
    print("Process time: ", end_time - start_time)

    # save the output images
    output_images = [
        Image.fromarray((output_image * 255).round().astype("uint8")) for output_image in output_images.as_numpy("IMAGES")
    ]
    image_path = []
    for idx, e_image in enumerate(output_images):
        e_image.save(
            artifacts + f'/image_{idx}.jpg'
        )
        image_path.append(f'artifacts/image_{idx}.jpg')

    return JSONResponse(
        content=image_path, status_code=200
    )



###################
# Helper functions
###################

def requestGenerator(inputs_data: List[Any], input_names, output_names, dtypes):
    # define protocol
    if protocol.lower() == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input and output data
    inputs = [
        client.InferInput(e_input_name, e_input.shape, e_dtype).set_data_from_numpy(e_input)
        for e_input_name, e_input, e_dtype in zip(input_names, inputs_data, dtypes)
    ]
    outputs = [
        client.InferRequestedOutput(output_name)
        for output_name in output_names
    ]
    return inputs, outputs