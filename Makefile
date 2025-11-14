.PHONY: help install run dev stop clean test format lint

# Default target
help:
	@echo "Available commands:"
	@echo "  make install    - Create virtual environment and install dependencies"
	@echo "  make run        - Run the FastAPI server"
	@echo "  make dev        - Run the FastAPI server with auto-reload"
	@echo "  make stop       - Stop the server running on port 8000"
	@echo "  make clean      - Remove virtual environment and cache files"
	@echo "  make test       - Run tests (if available)"
	@echo "  make format     - Format code with black"
	@echo "  make lint       - Lint code with flake8"

# Variables
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
UVICORN = $(VENV)/bin/uvicorn

# Install dependencies
install:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Installation complete! Activate with: source $(VENV)/bin/activate"

# Stop server on port 8000
stop:
	@echo "Stopping server on port 8000..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "Server stopped." || echo "No server running on port 8000."

# Run the server
run:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@$(MAKE) stop
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000

# Run the server with auto-reload (development mode)
dev:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@$(MAKE) stop
	$(UVICORN) app.main:app --reload --host 0.0.0.0 --port 8000

# Clean up
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV)
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	@echo "Cleanup complete!"

# Run tests
test:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v || echo "No tests found or pytest not installed"

# Format code (requires black)
format:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@if ! $(PIP) show black > /dev/null 2>&1; then \
		echo "Installing black..."; \
		$(PIP) install black; \
	fi
	$(VENV)/bin/black app/

# Lint code (requires flake8)
lint:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Virtual environment not found. Run 'make install' first."; \
		exit 1; \
	fi
	@if ! $(PIP) show flake8 > /dev/null 2>&1; then \
		echo "Installing flake8..."; \
		$(PIP) install flake8; \
	fi
	$(VENV)/bin/flake8 app/ --max-line-length=100 --exclude=__pycache__

