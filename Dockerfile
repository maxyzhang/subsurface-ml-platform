FROM python:3.12-slim

WORKDIR /app

ENV SUBSURFACE_ML_PROJECT_ROOT=/app

COPY pyproject.toml ./
COPY README.md ./

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY models ./models
COPY data ./data

RUN pip install --upgrade pip
RUN pip install .

EXPOSE 8000

CMD ["uvicorn", "subsurface_ml.api:app", "--host", "0.0.0.0", "--port",  "8000"]