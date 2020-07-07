#!/usr/bin/env python

"""
Tests for the xsec command-line scripts.

Run by executing, in the main repo directory:
    pytest
    pytest --no_download    (requires "gprocs" directory with gg data)
"""

import re
import subprocess
import pytest


def float_after(search_string, text):
    """
    Find a float value in terminal output, preceded by a specific search
    string.

    Parameters
    ----------
    search_string : str
        String preceding the sought number(s).
    text : str
        Complete input to parse and check.

    Returns
    -------
    float_list : list of float
        List of floats found occurring after search_string in text.
    """
    # Search for search_string, possibly followed by : or =, and/or brackets
    matches = re.findall(
        r"(?<={0})[:,=]?\s*\[?\s*[+-]?\d+\.\d*[Ee]?[+-]?\d*\]?".format(
            # r"(?<={0})[:,=]?\s*[+-]?\d+\.\d*[Ee]?[+-]?\d*".format(
            re.escape(search_string)
        ),
        text.decode(),
    )
    assert len(matches) == 1, "Multiple search string matches"
    match = matches[0]
    float_val = float(re.search(r"\[(.*?)\]", match).group(1))
    return float_val


def check_within_reasonable_range(output, xstype_string, min_value, max_value):
    """
    Check whether the cross-section or error value in the terminal
    output is within an expected range.

    Parameters
    ----------
    output : str
        Terminal output from xsec.
    xstype_string : str
        String preceding the desired numerical value in the output.
    min_value : float
        Minimum expected value.
    max_value : float
        Maximum expected value.

    Returns
    -------
    is_within_range : bool
        True if the specified value is within the specified bounds
        (including the boundary values), False otherwise.
    """
    assert xstype_string in output.decode()
    numerical_value = float_after(xstype_string, output)
    is_within_range = min_value <= numerical_value <= max_value
    return is_within_range


@pytest.fixture(scope="module")
def test_download_gg(pytestconfig):
    """
    Test the GP data file download for gluino pair production.
    """
    if not pytestconfig.getoption("no_download"):
        subprocess.check_call(["scripts/xsec-download-gprocs", "-t", "gg"])


def test_evaluation_gg(test_download_gg):
    """
    Run a test evaluation of the gluino pair production xsection.
    """
    # Collect terminal output
    output = subprocess.check_output(["scripts/xsec-test"])

    # Check whether output is in reasonable range
    assert check_within_reasonable_range(
        output, "xsection_central (fb): ", 245.0, 255.0
    ), "xsection_central outside expected range"
    assert check_within_reasonable_range(
        output, "regdown_rel", -0.05, -0.005
    ), "regdown_rel outside expected range"
    assert check_within_reasonable_range(
        output, "regup_rel", 0.005, 0.05
    ), "regup_rel outside expected range"
    assert check_within_reasonable_range(
        output, "scaledown_rel", -0.20, -0.05
    ), "scaledown_rel outside expected range"
    assert check_within_reasonable_range(
        output, "scaleup_rel", 0.05, 0.20
    ), "scaleup_rel outside expected range"
    assert check_within_reasonable_range(
        output, "pdfup_rel", 0.05, 0.15
    ), "pdfup_rel outside expected range"
    assert check_within_reasonable_range(
        output, "alphasup_rel", 0.005, 0.05
    ), "alphasup_rel outside expected range"

    # Check symmetric errors
    assert float_after("pdfdown_rel", output) == -float_after(
        "pdfup_rel", output
    )
    assert float_after("alphasdown_rel", output) == -float_after(
        "alphasup_rel", output
    )
