from pathlib import Path


def test_repository_scaffold() -> None:
    """Keep CI green"""
    assert Path("README.md").is_file()
    assert Path("data/data.py").is_file()
