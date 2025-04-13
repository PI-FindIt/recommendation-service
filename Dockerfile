FROM python:3.12
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV UV_VIRTUALENVS_CREATE false
ENV UV_VIRTUALENVS_IN_PROJECT false
ENV ENV development
ENV TORCH_CUDA_VERSION="cpu"
ENV PATH="$PATH:/root/.local/bin"


WORKDIR /recommendation-service

RUN pip install --no-cache uv uvicorn

COPY uv.lock pyproject.toml ./
RUN uv sync --group dev

EXPOSE 8000
CMD [ "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload" ]
