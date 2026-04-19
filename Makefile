.PHONY: install install-dev train rag test lint format typecheck clean run

# ─── Setup ───────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# ─── ML Pipeline ─────────────────────────────────────────────────

train:
	python src/train.py

rag:
	python src/build_rag.py

eda:
	python notebooks/eda.py

# ─── Development ─────────────────────────────────────────────────

run:
	streamlit run app.py

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint:
	flake8 src/ app.py tests/

format:
	black src/ app.py tests/
	isort src/ app.py tests/

typecheck:
	mypy src/

# ─── Cleanup ─────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache htmlcov .coverage
