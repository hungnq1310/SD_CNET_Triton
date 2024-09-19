### Pip stage ###

FROM python:3.10-slim as compiler
ENV PYTHONUNBUFFERED 1

ADD . .

RUN python -m venv /venv

ENV PATH="/venv/bin:$PATH"

RUN pip install nvidia-cuda-runtime-cu11 && \
    pip install fastapi[standard] uvicorn python-multipart attrdict && \
    pip install tritonclient[all] pillow numpy==1.26 xformers grpcio python-dotenv

FROM python:3.10-slim as runner

COPY --from=compiler /venv /venv
COPY src /src

ENV PATH="/venv/bin:$PATH"

CMD fastapi run src/api.py --port 6999