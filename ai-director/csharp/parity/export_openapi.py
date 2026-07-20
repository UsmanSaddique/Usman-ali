"""
Export the FastAPI OpenAPI schema to csharp/parity/fixtures/openapi.json.

This is the API-compatibility contract: the C# minimal-API endpoints must match
these routes, methods, and schema field names so frontend/index.html keeps
working. Run on the same python that runs the app (has fastapi installed):

    python csharp/parity/export_openapi.py
"""
import json
import sys
from pathlib import Path

# Repo root on sys.path so "import app.main" works.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT = Path(__file__).parent / "fixtures" / "openapi.json"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    from app.main import app  # FastAPI instance
    schema = app.openapi()
    OUT.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    paths = schema.get("paths", {})
    print(f"Wrote {OUT} — {len(paths)} routes")


if __name__ == "__main__":
    main()
