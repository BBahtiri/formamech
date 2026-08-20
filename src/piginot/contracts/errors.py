"""Typed failures raised at PI-GINOT contract boundaries."""


class ContractValidationError(ValueError):
    """Base class for invalid contract data."""


class BareDimensionalValueError(ContractValidationError):
    """A dimensional field was supplied without an explicit unit."""


class NonFiniteNumberError(ContractValidationError):
    """A contract contained NaN or an infinity."""


class UnsupportedUnitError(ContractValidationError):
    """A quantity used a unit outside the finite supported registry."""


class UnitMismatchError(ContractValidationError):
    """A quantity or expression has the wrong physical dimension."""


class ExpressionSyntaxError(ContractValidationError):
    """A derived expression used syntax outside the safe grammar."""


class UnknownSymbolError(ContractValidationError):
    """A derived expression referenced an undeclared parameter."""


class DuplicateKeyError(ContractValidationError):
    """An inbound JSON object repeated a member name."""


class InvalidHashDomainError(ContractValidationError):
    """A hash domain was not one of the four public domains."""


class UnknownFieldError(ContractValidationError):
    """A mapping contained a field absent from the target contract."""
