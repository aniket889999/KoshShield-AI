.PHONY: bootstrap dev-api dev-web infra-up infra-down test lint generate-key

bootstrap:
	/opt/anaconda3/bin/python3.12 -m venv .venv
	.venv/bin/pip install -e "apps/api[dev]"
	pnpm install

dev-api:
	.venv/bin/uvicorn koshshield.main:app --app-dir apps/api/src --reload --port 8000

dev-web:
	pnpm --filter @koshshield/web dev

infra-up:
	docker compose up -d postgres qdrant

infra-down:
	docker compose down

test:
	.venv/bin/pytest apps/api/tests
	pnpm --filter @koshshield/web test

lint:
	.venv/bin/ruff check apps/api
	.venv/bin/ruff format --check apps/api
	pnpm --filter @koshshield/web lint

generate-key:
	/opt/anaconda3/bin/python3.12 scripts/generate_master_key.py
