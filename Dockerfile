FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/mlx-repro
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY artifacts/targets ./artifacts/targets
COPY tests ./tests
RUN python -m pip install --no-cache-dir -e '.[dev]' && pytest -q

ENTRYPOINT ["mlxsim"]
CMD ["--help"]
