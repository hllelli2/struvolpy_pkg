import importlib


def requires_TEMPy(func):
    """
    A decorator that checks if the TEMPy package is installed before running a function.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The decorated function.

    Raises:
        ImportError: If the TEMPy package is not installed.

    Example:
        >>> @requires_TEMPy
        ... def my_function():
        ...     pass
    """

    def wrapper(*args, **kwargs):
        try:
            importlib.import_module("TEMPy")
        except ImportError:
            raise ImportError(
                "TEMPy is required to use this. "
                "Please install it using `pip install biotempy "
                "or pip install struvolpy[tempy]"
            )
        return func(*args, **kwargs)

    return wrapper
