"""Run using: python -m software.backend (inside tmux)."""
import os


def main():
    if not os.environ.get("TMUX"):
        raise SystemExit("Start the service inside tmux; see software/docs/deployment.md")
    import uvicorn
    from software.backend.app import create_app
    uvicorn.run(create_app(), host=os.environ.get("SEG_HOST", "127.0.0.1"),
                port=int(os.environ.get("SEG_PORT", "8000")), workers=1, proxy_headers=False)


if __name__ == "__main__":
    main()
