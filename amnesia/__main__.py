"""Allow ``python -m amnesia`` as an alternative to the ``amnesia`` console script."""

from __future__ import annotations

from amnesia.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
