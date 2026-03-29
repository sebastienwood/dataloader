# ── dino-datasets justfile ────────────────────────────────────────────────────
#
# Usage:
#   just                                  list all recipes (default)
#   just rename my_datasets               preview the package rename (dry-run)
#   just rename my_datasets --apply       apply the rename
#   just rename my_datasets --apply --from interim_name

# Show available recipes by default.
default:
    @just --list

# ── Variables ─────────────────────────────────────────────────────────────────

from   := "dino_loader"

# ── Package rename ────────────────────────────────────────────────────────────

# Preview or apply a package rename.
# NEW is the target snake_case name (e.g. my_datasets).
# Pass --apply as extra to execute; dry-run otherwise.
# Pass --from <old> to override the current package name (default: dino_loader).
#
# Examples:
#   just rename my_datasets
#   just rename my_datasets --apply
#   just rename my_datasets --apply --from interim_name
rename new *extra:
    uv run python scripts/rename.py {{ new }} --from {{ from }} {{ extra }}

# ── Development ───────────────────────────────────────────────────────────────

# Install the package in editable mode with dev extras, then regenerate stubs.
# Always run this instead of bare `uv sync` so stubs stay in sync with the
# installed versions of dino_datasets and dino_env.
install:
    uv sync --all-groups
    just gen-stubs

# Regenerate typed stubs for dino_datasets and dino_env from the installed
# packages.  Run after `uv sync` whenever those packages change.
gen-stubs:
    uv run python scripts/gen_stubs.py

# Check that committed stubs match the installed packages (used in CI).
check-stubs:
    uv run python scripts/gen_stubs.py --check

# Run the full test suite.
test:
    uv run pytest

# Run ruff linter and format checker.
lint:
    uvx ruff check src/ scripts/ --fix
    uvx ruff format --check src/ tests/ scripts/

# Run ty static type checker on src/.
typecheck:
    uvx ty check src/
