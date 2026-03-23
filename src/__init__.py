from typing import Any

__all__ = ["OracleKVProjectAPI"]


def __getattr__(name: str) -> Any:
	if name == "OracleKVProjectAPI":
		from .api import OracleKVProjectAPI

		return OracleKVProjectAPI
	raise AttributeError(f"module 'src' has no attribute {name!r}")
