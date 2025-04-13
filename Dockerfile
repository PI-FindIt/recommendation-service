FROM python:3.12
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV POETRY_VIRTUALENVS_CREATE false
ENV POETRY_VIRTUALENVS_IN_PROJECT false
ENV ENV development

WORKDIR /recommendation-service

RUN pip install --no-cache poetry

COPY poetry.lock pyproject.toml ./
RUN poetry install --with dev

EXPOSE 8000
CMD [ "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload" ]
