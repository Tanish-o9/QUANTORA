.PHONY: help dev test api web install docker lint

help:
	@echo "Quantora Commands:"
	@echo "  make dev      - Start API backend and React UI simultaneously"
	@echo "  make test     - Run full PyTest test suite"
	@echo "  make api      - Run Python REST API server"
	@echo "  make web      - Run React Vite dev server"
	@echo "  make install  - Install Python dependencies and CLI globally (editable)"
	@echo "  make docker   - Build and run with Docker Compose"

install:
	pip install -e .
	cd web && npm install

api:
	python server.py

web:
	cd web && npm run dev

test:
	python -m pytest tests/

dev:
	@echo "Starting API and Web UI concurrently..."
	python server.py &
	cd web && npm run dev

docker:
	docker compose up --build
