"""
Pytest configurations to be shared among all tests.
"""


def pytest_addoption(parser):
    """
    Add config options when running pytest.
    """
    parser.addoption(
        "--no_download",
        action="store_true",
        help="Turn off GP download test",
    )
