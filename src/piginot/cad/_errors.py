"""Explicit CAD input and capability refusals."""


class CadImportError(ValueError):
    """An input cannot be interpreted as the supported planar profile."""

    def __init__(self, rule: str, message: str) -> None:
        self.rule = rule
        super().__init__(f"[{rule}] {message}")
