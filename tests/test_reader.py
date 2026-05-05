"""
Test the reader.
"""

from pathlib import Path

import pytest
from numpy.testing import assert_allclose

from qres_reader import read_qres_file, read_qres_file_as_dataframe

TEST_FILE = Path("tests/data/Data_Test.qres")


def test_reader():
    header, data = read_qres_file(TEST_FILE)

    # Test header data
    assert header["File version"] == 2
    assert header["Sample rate"] == 100.0
    assert header["Number of channels"] == 7
    assert header["Number of samples"] == 3104
    assert header["Units"] == ["kN", "mm", "mm", "mm", "mm", "%", "mm", "s"]

    # Test actual data
    assert_allclose([8.20079632e-02, 3.11457726e01], data[0][:2])
    assert_allclose([0.07602564, 31.14556638], data[6][:2])


def test_reader_df():
    data = read_qres_file_as_dataframe(TEST_FILE)

    assert all(
        data.columns
        == [
            "Axial Kraft [kN]",
            "Axial Weg [mm]",
            "MT10-1 Weg [mm]",
            "MT10-2 Weg [mm]",
            "Mittelwert Weg [mm]",
            "Axial  Servoventilbefehl [%]",
            "Axial Command [mm]",
        ]
    )


if __name__ == "__main__":
    pytest.main()
