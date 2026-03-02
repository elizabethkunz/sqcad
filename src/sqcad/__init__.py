# src/sqcad/__init__.py

"""sqCAD: superconducting quantum circuit analysis & design."""

from __future__ import annotations

__all__ = ["__version__", "about"]


def about() -> str:
    return "sqCAD (Superconducting Quantum Circuit Analysis & Design)"