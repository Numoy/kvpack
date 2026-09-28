# kvpack server image. Linux PyTorch wheels from PyPI include CUDA, so this runs on
# NVIDIA GPUs (with the NVIDIA container toolkit) and falls back to CPU elsewhere.
#
#   docker build -t kvpack .
#   docker run --gpus all -p 8000:8000 -v $PWD/cartridges:/cartridges \
#     -e KVPACK_API_KEYS=change-me kvpack

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" HF_HOME=/root/.cache/huggingface

WORKDIR /app
# Dependencies first, so code changes don't reinstall PyTorch.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

VOLUME ["/cartridges", "/root/.cache/huggingface"]
EXPOSE 8000
ENTRYPOINT ["kvpack"]
CMD ["serve", "/cartridges", "--host", "0.0.0.0", "--port", "8000"]
