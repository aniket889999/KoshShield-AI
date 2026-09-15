.PHONY: bootstrap dev-api dev-web infra-up infra-down tool-runner-build test lint generate-key migrate runtime-preflight smoke-local provision-runtime
.PHONY: test-api test-web lint-api lint-web dependency-inventory prepare-dependencies

WHEELHOUSE ?= data/wheelhouse

bootstrap:
	/opt/anaconda3/bin/python3.12 -m venv .venv
	.venv/bin/pip install -e "apps/api[dev]"
	pnpm install

migrate:
	.venv/bin/python -m koshshield.database.migration

dev-api:
	.venv/bin/uvicorn koshshield.main:app --app-dir apps/api/src --reload --port 8000

dev-web:
	pnpm --filter @koshshield/web dev

infra-up:
	docker compose up -d postgres qdrant

infra-down:
	docker compose down

tool-runner-build:
	docker compose --profile tools build tool-runner

test: test-api test-web

test-api:
	.venv/bin/pytest apps/api/tests

test-web:
	pnpm --filter @koshshield/web test

lint: lint-api lint-web

lint-api:
	.venv/bin/ruff check apps/api scripts
	.venv/bin/ruff format --check apps/api scripts

lint-web:
	pnpm --filter @koshshield/web lint

generate-key:
	/opt/anaconda3/bin/python3.12 scripts/generate_master_key.py

runtime-preflight:
	.venv/bin/python -m koshshield.runtime_preflight

smoke-local:
	.venv/bin/python -m koshshield.smoke

provision-runtime:
	.venv/bin/python scripts/provision_local_runtime.py --dry-run

dependency-inventory:
	.venv/bin/python scripts/prepare_runtime_dependencies.py --wheelhouse "$(WHEELHOUSE)"

prepare-dependencies:
	.venv/bin/python scripts/prepare_runtime_dependencies.py --wheelhouse "$(WHEELHOUSE)" --prepare
