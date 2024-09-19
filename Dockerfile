### Pip stage ###

FROM python:3.10-slim as compiler
ENV PYTHONUNBUFFERED 1

ADD . .

RUN python -m venv /venv

ENV PATH="/venv/bin:$PATH"

RUN pip install nvidia-cuda-runtime-cu11 && \
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install fastapi[standard] uvicorn python-multipart attrdict && \
    pip install tritonclient[all] transformers diffusers pillow numpy==1.26 xformers

FROM python:3.10-slim as runner

COPY --from=compiler /venv /venv
COPY src /src

ENV PATH="/venv/bin:$PATH"

CMD fastapi run src/api.py --port 7999