.PHONY: up down build test seed lint lint-back lint-front lint-forecasting test-forecasting

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

test:
	docker compose exec backend pytest -q

seed:
	docker compose exec backend python scripts/seed_ipc.py

lint: lint-back lint-front lint-forecasting

lint-forecasting:
	ruff check shared tfg-forecasting/tests

test-forecasting:
	pytest tfg-forecasting/tests/test_metrics.py -q

lint-back:
	cd tfg-arquitectura/backend && ruff check app scripts tests

lint-front:
	cd tfg-arquitectura/frontend && npm run lint
