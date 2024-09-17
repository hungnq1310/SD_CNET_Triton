import inspect
from typing import Callable, List, Optional, Union, Dict
import logging

# noinspection DuplicatedCode
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPFeatureExtractor, CLIPTokenizer

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, PIL_INTERPOLATION
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE


try:
    # noinspection PyUnresolvedReferences
    import triton_python_backend_utils as pb_utils
except ImportError:
    pass  # triton_python_backend_utils exists only inside Triton Python backend.

class TritonPythonModel:
    # vae_encoder: OnnxRuntimeModel
    # vae_decoder: OnnxRuntimeModel
    # text_encoder: OnnxRuntimeModel
    # unet: OnnxRuntimeModel
    # controlnet: OnnxRuntimeModel

    tokenizer: CLIPTokenizer
    device: str
    scheduler: Union[
        DDIMScheduler,
        PNDMScheduler,
        LMSDiscreteScheduler,
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
    ]
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    eta: float
    
    def initialize(self, args: Dict[str, str]) -> None:
        
        current_name: str = str(Path(args["model_repository"]).parent.absolute())
        
        if args.get("model_instance_kind") == "GPU":
            self.device = "cuda"
        else: 
            self.device = "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained(
            current_name + "/stable_diffusion_cnet_bls/1/tokenizer/"
        )
        self.scheduler_config_path = current_name + "/stable_diffusion_cnet_bls/1/scheduler/"
        self.scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler_config_path)

        self.height:int = 512
        self.width:int = 512
        # self.num_inference_steps:int = 50
        self.guidance_scale:float = 7.5
        self.eta:float = 0.0
        self.callback_steps:int = 1
        self.callback: Callable[[int, int, np.ndarray], None] = None
        self.output_type:str = "pil"
        self.num_images_per_prompt:int = 1
        self.num_inference_steps: int= 50
        self.controlnet_conditioning_scale: float = 1.0
        self.logger = pb_utils.Logger
        

        
        
# UTILS UTILS UTILS UTILS UTILS UTILS UTILS UTILS UTILS UTILS
# UTILS UTILS UTILS UTILS UTILS UTILS UTILS UTILS UTILS UTILS
# UTILS UTILS UTILS UTILS UTILS UTILS UTILS UTILS UTILS UTILS

    # def _default_height_width(self, height, width, image):
    #     if isinstance(image, list):
    #         image = image[0]

    #     if height is None:
    #         if isinstance(image, PIL.Image.Image):
    #             height = image.height
    #         elif isinstance(image, np.ndarray):
    #             height = image.shape[3]

    #         height = (height // 8) * 8  # round down to nearest multiple of 8

    #     if width is None:
    #         if isinstance(image, PIL.Image.Image):
    #             width = image.width
    #         elif isinstance(image, np.ndarray):
    #             width = image.shape[2]

    #         width = (width // 8) * 8  # round down to nearest multiple of 8

    #     return height, width
        
    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt, dtype):
        # print(image)
        # if not isinstance(image, torch.Tensor):
        #     if isinstance(image, PIL.Image.Image):
        #         image = [image]

        #     # if isinstance(image[0], PIL.Image.Image):
        #         image = [
        #             torch.tensor(np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])))[None, :]
        #             for i in image
        #         ]
        #         image = torch.cat(image, dim=0)
        #         image = image.float() / 255.0
        #         image = image.permute(0, 3, 1, 2)  # Transpose for PyTorch (batch, channels, height, width)
        #     elif isinstance(image, List[torch.Tensor]):
        #         image = torch.cat(image, dim=0)
                
        #     elif isinstance(image, np.array):
        #         image = torch.tensor(image)
        #         image = image.float() / 255.0
        #         image = image.permute(0, 3, 1, 2)  

        
        #* image pass through triton server is np.array
        image = torch.tensor(image)
        # image = image.resize((width, height))
        # image = torch.cat(image, dim=0)
        image = image.unsqueeze(0)
        image = image.float() / 255.0
        image = image.permute(0, 3, 1, 2)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # Image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)
        
        if image.dtype != dtype:
            image = image.to(dtype)
                
        return image #torch.float32
        
        
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        
        # Check if generator is a PyTorch generator
        if not isinstance(generator, torch.Generator):
            raise TypeError("The generator should be an instance of torch.Generator.")
        
        if latents is None:
            # Generate latents using torch.randn with the provided generator
            latents = torch.randn(*shape, generator=generator, dtype=dtype)

        # Scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        
        return latents
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta, torch_gen):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = torch_gen
        return extra_step_kwargs

    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # get prompt text embeddings with truncation
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        ).input_ids.astype(np.int32)
        # raw prompt without truncation
        untruncated_ids = self.tokenizer(
            prompt, 
            padding="max_length", 
            return_tensors="np"
        ).input_ids.astype(np.int32)

        # only for logging
        if not np.array_equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            self.logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            
        # call `text encoder` for prompt embeddings (have truncation) 
        # input_ids = text_input_ids.type(dtype=torch.int32)
        text_input_encoder = pb_utils.Tensor("input_ids", text_input_ids)
        
        inference_request = pb_utils.InferenceRequest(
            model_name="text_encoder",
            requested_output_names=["last_hidden_state"],
            inputs=[text_input_encoder],
        )
        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(
                inference_response.error().message()
            )
        
        output = pb_utils.get_output_tensor_by_name(
            inference_response, "last_hidden_state"
        )
        text_embeddings: torch.Tensor = torch.from_dlpack(output.to_dlpack())
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        do_classifier_free_guidance = self.guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        batch_size = 1
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input_ids = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            ).input_ids.astype(np.int32)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            uncond_input = pb_utils.Tensor("input_ids", uncond_input_ids)

            inference_request = pb_utils.InferenceRequest(
                model_name="text_encoder",
                requested_output_names=["last_hidden_state"],
                inputs=[uncond_input],
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, "last_hidden_state"
                )
                uncond_embeddings: torch.Tensor = torch.from_dlpack(
                    output.to_dlpack()
                )

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(
                1, num_images_per_prompt, 1
            )
            uncond_embeddings = uncond_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings
        
        
        
         
    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        
        responses = []
        # for loop for batch requests (minh` xai` batch 0 nen loop cho nay` skip dc`)
        for request in requests:
            # client send binary data typed - convert back to string
            prompt = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "PROMPT")
                .as_numpy()
                .tolist()
            ]
            
            negative_prompt = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "NEGATIVE_PROMPT")
                .as_numpy()
                .tolist()
            ]
            
            pose_image = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "POSE_IMAGE")
                .as_numpy()
                .tolist()
            ]
            
            scheduler = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "SCHEDULER")
                .as_numpy()
                .tolist()
            ][0]
            if scheduler.__class__.__name__ != scheduler:
                self.scheduler = eval(
                    f"{scheduler}.from_config(self.scheduler_config_path)"
                )
                
            self.num_inference_steps = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "STEPS")
                .as_numpy()
                .tolist()
            ][0]
            
            self.guidance_scale = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "GUIDANCE_SCALE")
                .as_numpy()
                .tolist()
            ][0]
            
            seed = [
                t
                for t in pb_utils.get_input_tensor_by_name(request, "SEED")
                .as_numpy()
                .tolist()
            ][0]
            
            if isinstance(prompt, str):
                batch_size = 1
            elif isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
            
            # if generator:
            #     torch_seed = generator.randint(seed)
            #     torch_gen = torch.Generator().manual_seed(torch_seed)
            # else:
            generator = torch.Generator().manual_seed(seed)
            torch_gen = None
                
            # height = 512
            # width = 512        
            if self.height % 8 != 0 or self.width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {self.height } and {self.width}.")
            if (self.callback_steps is None) or (
                self.callback_steps is not None and (not isinstance(self.callback_steps, int) or self.callback_steps <= 0)
            ):
                raise ValueError(
                    f"`self.callback_steps` has to be a positive integer but is {self.callback_steps} of type"
                    f" {type(self.callback_steps)}."
                )
            
            # If negative_promt empty = None (handle error)
            if negative_prompt[0] == "NONE":
                negative_prompt = None
            
        do_classifier_free_guidance = self.guidance_scale > 1.0

        # encode prompt
        prompt_embeds: torch.Tensor = self._encode_prompt(
            prompt, self.num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        
        # 4. Prepare image
        image = self.prepare_image(
            image=pose_image,
            width=self.width,
            height=self.height ,
            batch_size=batch_size * self.num_images_per_prompt,
            num_images_per_prompt=self.num_images_per_prompt,
            dtype=torch.float32,
        )
        

        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        # latents_shape = (batch_size * self.num_images_per_prompt, 4, height // 8, width // 8)
        
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size=(batch_size * self.num_images_per_prompt),
            num_channels_latents= num_channels_latents,
            height=self.height ,
            width=self.width,
            dtype=latents_dtype,
            generator=generator,
        )

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2, dim=0)
            latents = torch.cat([latents] * 2)
        
        # set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps) #? why device put here
        timesteps = self.scheduler.timesteps


        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, self.eta, torch_gen)

        # prepare the output
        # timestep_dtype = ORT_TO_NP_TYPE["tensor(float)"]
        num_warmup_steps = len(timesteps) - self.num_inference_steps * self.scheduler.order
        
        with tqdm(total=self.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                timestep = t[None].type(dtype=torch.float32)
                encoder_hidden_states = prompt_embeds.type(dtype=torch.float32)
            
                # inputs_cnet = [
                #     pb_utils.Tensor.from_dlpack("sample", torch.to_dlpack(latent_model_input.type(dtype=torch.float16))),
                #     pb_utils.Tensor.from_dlpack("timestep", torch.to_dlpack(timestep)),
                #     pb_utils.Tensor.from_dlpack("encoder_hidden_states", torch.to_dlpack(encoder_hidden_states)),
                #     pb_utils.Tensor.from_dlpack("controlnet_cond", torch.to_dlpack(image.type(dtype=torch.float16))),
                #     pb_utils.Tensor.from_dlpack("conditioning_scale", torch.to_dlpack(torch.tensor([self.controlnet_conditioning_scale], dtype=torch.float16))),
                # ]

                inputs_cnet = [
                    pb_utils.Tensor("sample", latent_model_input.cpu().numpy()),
                    pb_utils.Tensor("timestep", timestep.cpu().numpy()),
                    pb_utils.Tensor("encoder_hidden_states", encoder_hidden_states.cpu().numpy()), 
                    pb_utils.Tensor("controlnet_cond", image.numpy()),
                    pb_utils.Tensor("conditioning_scale", 
                                               np.array([self.controlnet_conditioning_scale])
                                               ),
                ]
                
                # Truong model
                # outputs_cnet = [
                #     "output", "2587", "2588", "2589", "2590", 
                #     "2591", "2592", "2593", "2594", "2595", 
                #     "2596", "2597", "2598"
                # ]        
                       
                # new model
                outputs_cnet = [
                    "down_block_res_samples", "mid_block_res_sample", #! missing concept name
                    "2773", "2775", "2777", "2779", "2781", "2783", 
                    "2785", "2787", "2789", "2791", "2793"
                ]
                
                inference_request = pb_utils.InferenceRequest(
                    model_name="cnet",  # Assuming the model is named "controlnet"
                    requested_output_names=outputs_cnet,  # Replace with the actual output names you need
                    inputs=inputs_cnet,
                )
                # Send infer to controlnet
                inference_response = inference_request.exec()
                if inference_response.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response.error().message())
                else:
                    # Extract the output tensors from the inference response.
                    # mid_block_res_sample = pb_utils.get_output_tensor_by_name(inference_response, '2598')
                    # down_block_res_samples = (
                    #     pb_utils.get_output_tensor_by_name(inference_response, 'output'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2587'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2588'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2589'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2590'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2591'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2592'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2593'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2594'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2595'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2596'),
                    #     pb_utils.get_output_tensor_by_name(inference_response, '2597')
                    # )
                    # mid_block_res_sample = pb_utils.get_output_tensor_by_name(inference_response, '2793')
                    mid_block_res_sample = torch.from_dlpack(
                        pb_utils.get_output_tensor_by_name(inference_response, '2793')
                    )
                    down_block_res_samples = [
                        torch.from_dlpack(
                            pb_utils.get_output_tensor_by_name(inference_response, name).to_dlpack()
                        # pb_utils.get_output_tensor_by_name(inference_response, name).as_numpy()
                        ) 
                        for name in outputs_cnet[:-1]
                    ]
                down_block_res_samples = [
                    down_block_res_sample.to(device="cpu") * torch.tensor([self.controlnet_conditioning_scale])
                    for down_block_res_sample in down_block_res_samples
                ]
                mid_block_res_sample = mid_block_res_sample.to(device="cpu") * torch.tensor([self.controlnet_conditioning_scale])
                # mid_block_res_sample = mid_block_res_sample.as_numpy() * np.array([self.controlnet_conditioning_scale])
                            
                
                # inputs_unet = [
                #     pb_utils.Tensor.from_dlpack("sample", torch.to_dlpack(latent_model_input.type(dtype=torch.float32))),
                #     pb_utils.Tensor.from_dlpack("timestep", torch.to_dlpack(timestep.type(dtype=torch.float32))),
                #     pb_utils.Tensor.from_dlpack("encoder_hidden_states", torch.to_dlpack(prompt_embeds.type(dtype=torch.float32))),
                #     pb_utils.Tensor.from_dlpack("down_block_0", torch.to_dlpack(down_block_res_samples[0])),
                #     pb_utils.Tensor.from_dlpack("down_block_1", torch.to_dlpack(down_block_res_samples[1])),
                #     pb_utils.Tensor.from_dlpack("down_block_2", torch.to_dlpack(down_block_res_samples[2])),
                #     pb_utils.Tensor.from_dlpack("down_block_3", torch.to_dlpack(down_block_res_samples[3])),
                #     pb_utils.Tensor.from_dlpack("down_block_4", torch.to_dlpack(down_block_res_samples[4])),
                #     pb_utils.Tensor.from_dlpack("down_block_5", torch.to_dlpack(down_block_res_samples[5])),
                #     pb_utils.Tensor.from_dlpack("down_block_6", torch.to_dlpack(down_block_res_samples[6])),
                #     pb_utils.Tensor.from_dlpack("down_block_7", torch.to_dlpack(down_block_res_samples[7])),
                #     pb_utils.Tensor.from_dlpack("down_block_8", torch.to_dlpack(down_block_res_samples[8])),
                #     pb_utils.Tensor.from_dlpack("down_block_9", torch.to_dlpack(down_block_res_samples[9])),
                #     pb_utils.Tensor.from_dlpack("down_block_10", torch.to_dlpack(down_block_res_samples[10])),
                #     pb_utils.Tensor.from_dlpack("down_block_11", torch.to_dlpack(down_block_res_samples[11])),
                #     pb_utils.Tensor.from_dlpack("mid_block_additional_residual", torch.to_dlpack(mid_block_res_sample)),
                # ]
                inputs_unet = [
                    pb_utils.Tensor("sample", latent_model_input.cpu().numpy()),
                    pb_utils.Tensor("timestep", timestep.cpu().numpy()),
                    pb_utils.Tensor("encoder_hidden_states", prompt_embeds.cpu().numpy()),
                    pb_utils.Tensor("down_block_0", down_block_res_samples[0].numpy()),
                    pb_utils.Tensor("down_block_1", down_block_res_samples[1].numpy()),
                    pb_utils.Tensor("down_block_2", down_block_res_samples[2].numpy()),
                    pb_utils.Tensor("down_block_3", down_block_res_samples[3].numpy()),
                    pb_utils.Tensor("down_block_4", down_block_res_samples[4].numpy()),
                    pb_utils.Tensor("down_block_5", down_block_res_samples[5].numpy()),
                    pb_utils.Tensor("down_block_6", down_block_res_samples[6].numpy()),
                    pb_utils.Tensor("down_block_7", down_block_res_samples[7].numpy()),
                    pb_utils.Tensor("down_block_8", down_block_res_samples[8].numpy()),
                    pb_utils.Tensor("down_block_9", down_block_res_samples[9].numpy()),
                    pb_utils.Tensor("down_block_10", down_block_res_samples[10].numpy()),
                    pb_utils.Tensor("down_block_11", down_block_res_samples[11].numpy()),
                    pb_utils.Tensor("mid_block_additional_residual", mid_block_res_sample.numpy()),
                ]

                # Create the inference request for the `unet` model
                inference_request_unet = pb_utils.InferenceRequest(
                    model_name="unet",  # Name of the model
                    requested_output_names=["out_sample"],  # Output names as expected from `unet`
                    inputs=inputs_unet,
                )

                # Execute the inference request
                inference_response_unet = inference_request_unet.exec()

                if inference_response_unet.has_error():
                    raise pb_utils.TritonModelException(
                        inference_response_unet.error().message()
                    )
                else:
                    output = pb_utils.get_output_tensor_by_name(
                        inference_response_unet, "out_sample"
                    )
                    noise_pred: torch.Tensor = torch.from_dlpack(output.to_dlpack())
                
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred.to(device=self.device), t, latents.to(device=self.device), **extra_step_kwargs
                ).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if self.callback is not None and i % self.callback_steps == 0:
                        self.callback(i, t, latents)

            latents = 1 / 0.18215 * latents
            latents = latents.type(dtype=torch.float32)
            inputs = [
                pb_utils.Tensor.from_dlpack(
                    "latent_sample", torch.to_dlpack(latents)
                )
            ]
            inference_request = pb_utils.InferenceRequest(
                model_name="vae_decoder",
                requested_output_names=["sample"],
                inputs=inputs,
            )
            inference_response = inference_request.exec()
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message()
                )
            else:
                output = pb_utils.get_output_tensor_by_name(inference_response, "sample")
                image: torch.Tensor = torch.from_dlpack(output.to_dlpack())
                image = image.type(dtype=torch.float32)
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).numpy()
                
            tensor_output = [pb_utils.Tensor("IMAGES", image)]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses