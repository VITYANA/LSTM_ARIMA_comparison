from importlib import import_module


def test_data_module_is_importable() -> None:
    """Ensure the initial package structure is valid."""
    module = import_module("data.data")

    assert module.__name__ == "data.data"
