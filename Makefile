.PHONY: bootstrap test lint reproduce-architecture

bootstrap:
	./scripts/bootstrap.sh

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

reproduce-architecture:
	.venv/bin/mlxsim reproduce --figure all --output artifacts/results/architecture-latest.json
