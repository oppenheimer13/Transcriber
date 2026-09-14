"""Entry point for `python -m transcriber`, so the tool runs uninstalled."""

from .cli import run

if __name__ == "__main__":
    run()
