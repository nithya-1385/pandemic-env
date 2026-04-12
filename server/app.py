import sys
import os
from fastapi import FastAPI

# ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import your FastAPI app from server.py
from server import app


def main():
    """
    Entry point required by OpenEnv validator.
    This is what makes the server 'multi-mode deployable'.
    """
    import uvicorn

    port = int(os.environ.get("PORT", 7860))

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )


if __name__ == "__main__":
    main()