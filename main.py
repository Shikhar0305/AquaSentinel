"""
Start AquaSentinel:  python main.py

Opens the website and API on http://localhost:8000
(same as: python -m uvicorn api.main:app --port 8000)
"""
import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AquaSentinel API and website")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart automatically when code changes")
    args = parser.parse_args()
    print(f"AquaSentinel running on http://localhost:{args.port}  (API docs: /docs)")
    uvicorn.run("api.main:app", host="127.0.0.1", port=args.port, reload=args.reload)
