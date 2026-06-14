# forest-sounds task runner
# Recipes mirror the Python CI workflow (.github/workflows/python.yml)

# Install deps + dev deps
install:
    uv sync --all-extras --dev

# Check formatting (no writes)
fmt-check:
    uv run ruff format --check . 
# Auto-format
fmt:
    uv run ruff format . 

# Lint
lint:
    uv run ruff check . --preview

fix:
    uv run ruff check . --preview --fix

# Run tests
test:
    uv run pytest

# Full CI: what runs in build_and_test
ci: fmt-check lint test
