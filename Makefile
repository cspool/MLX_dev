.PHONY: bootstrap test lint reproduce-architecture audit-completion

bootstrap:
	./scripts/bootstrap.sh

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check .

reproduce-architecture:
	.venv/bin/mlxsim reproduce --figure all --output artifacts/results/architecture-latest.json

audit-completion:
	.venv/bin/python scripts/audit_full_paper_completion.py --verify-existing
