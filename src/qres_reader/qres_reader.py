"""
Module to read *.qres files.

Usage
-----

Read a Cubus time history file:

    >>> from reader_qres import read_cubus_file
    >>> header, data = read_qres_file('file.qres')
    >>> print(header)
    { 'Date time': datetime.datetime(2018, 7, 4, 0, 0),
    'Description': ['Axial Kraft',
                    'Axial Weg',
                    'MT10-1 Weg',
                    'MT10-2 Weg',
                    'Mittelwert Weg',
                    'Axial  Servoventilbefehl',
                    'Axial Command',
                    'Time'],
    'File version': 2,
    'Number of channels': 7,
    'Number of samples': 3104,
    'Sample rate': 100.0,
    'Units': ['kN', 'mm', 'mm', 'mm', 'mm', '%', 'mm', 's']}

    >>> print(data)
    [[ 8.20079632e-02  3.11457726e+01 -4.87341377e-03 ...  1.51927042e+00
    3.11473668e+01  0.00000000e+00]
    ...
    [ 2.95137676e+01  3.21411216e+01 -4.85837680e-03 ... -1.17540997e-01
    3.20893809e+01  3.10300000e+01]]

A function to read the Cubus time history file directly into a pandas DataFrame
is provided, too:

    >>> from reader_qres import read_qres_file_as_dataframe
    >>> df_data = read_qres_file_as_dataframe('file.qres')
    >>> print(df_data)
            Axial Kraft  [kN]  Axial Weg  [mm]  MT10-1 Weg  [mm
    Time [s]
    0.00               0.082008        31.145773         -0.00487
    0.01               0.082008        31.145773         -0.00487
    0.02               0.081926        31.145766         -0.00487
    0.03               0.080565        31.145685         -0.00485
    0.04               0.076965        31.145545         -0.00482

"""

import datetime
import io
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def read_qres_file(
    file_name: str | Path, sample_rate: int | None = None
) -> tuple[dict, NDArray]:
    """Open Cubus time history file

    Parameters
    ----------
    file_name : str | Path
        Path to the file.
    sample_rate : int (optional)
        Get a different sample rate. (Not implemented)

    Returns
    -------
    header : dict
        Dictionary containing the header information.

    data : array
        Array with the data.

    """
    cubus_id_string = "CaTs3 Time History File"
    n_bytes = len(cubus_id_string) * 2

    # Check if the file is a Cubus time history file
    if not check_file_type(file_name):
        raise LookupError("Not a Cubus time history file")

    if sample_rate:
        raise NotImplementedError(
            "Defining a different sample rate is not implemented."
        )

    with io.open(file_name, "rb") as f:
        # Take first bytes out (containing file type information)
        f.read(n_bytes)

        # Get file version and date information
        file_version = np.fromfile(f, dtype="int32", count=1)[0]
        day = np.fromfile(f, dtype="int32", count=1)[0]
        month = np.fromfile(f, dtype="int32", count=1)[0]
        year = np.fromfile(f, dtype="int32", count=1)[0]
        hour = 0
        minute = 0
        seconds = 0
        # Parse date information
        current_date_time = datetime.datetime(year, month, day, hour, minute, seconds)

        if file_version > 1:
            is_cycles = np.fromfile(f, dtype="int8", count=1)[0]
        else:
            is_cycles = 0

        # Get number of sessions
        number_of_sessions = np.fromfile(f, dtype="int32", count=1)[0]

        if number_of_sessions != 1:
            raise NotImplementedError("Cannot open multi-session files")

        # Read the header for the one and only session
        session_index = np.fromfile(f, dtype="int32", count=1)[0]
        sample_rate = np.fromfile(f, dtype="double", count=1)[0]
        channel_count = np.fromfile(f, dtype="int32", count=1)[0]
        number_of_samples = np.fromfile(f, dtype="int32", count=1)[0]

        list_channel_names = []
        # Get the channel names
        for _ in range(channel_count):
            dum_desc = np.fromfile(f, dtype="int8", count=128 * 2)
            # Initialize empty string for the channel description
            ch_desc = ""
            for ix in range(128):
                ch_desc += chr(dum_desc[(ix * 2)])
            # Strip empty spaces
            list_channel_names.append(ch_desc.strip("\0").rstrip(" "))

        list_channel_units = []
        # Get the channel units
        for _ in range(channel_count):
            dum_units = np.fromfile(f, dtype="int8", count=20 * 2)
            # Initialize empty string for the channel units
            ch_units = ""
            for ix in range(20):
                ch_units += chr(dum_units[(ix * 2)])
            # Strip empty spaces
            list_channel_units.append(ch_units.strip("\0"))

        start_data_index = np.fromfile(f, dtype="int32", count=1)[0]
        end_data_index = np.fromfile(f, dtype="int32", count=1)[0]

        # Obtain the data from the channels
        if is_cycles == 0:
            # Initialize empty array to store the data
            # (An extra channel is added for the time)
            data = np.zeros((int(number_of_samples), int(channel_count) + 1))
            # Read data
            for s in range(number_of_samples):
                data[s, :-1] = np.fromfile(f, dtype="double", count=channel_count)
        else:
            # Initialize empty array to store the data
            # (An extra channel is added for the time and another for
            # the cycle information)
            data = np.zeros((int(number_of_samples), int(channel_count) + 2))
            # Read data
            for s in range(number_of_samples):
                data[s, :-1] = np.fromfile(f, dtype="double", count=(channel_count + 1))

        # Add a time channel
        time_end = number_of_samples / sample_rate
        time = np.linspace(
            0, time_end, int(np.round(time_end * sample_rate)), endpoint=True
        )
        data[:, -1] = time
        list_channel_names.append("Time")
        list_channel_units.append("s")

        # Create dictionary for the header
        header = {
            "File version": file_version,
            "Date time": current_date_time,
            "Sample rate": sample_rate,
            "Number of channels": channel_count,
            "Number of samples": number_of_samples,
            "Description": list_channel_names,
            "Units": list_channel_units,
        }

    return header, data


def read_qres_file_as_dataframe(file_name: str | Path) -> pd.DataFrame:
    """Read cubus time history file as a DataFrame

    Parameters
    ----------
    file_name : str | Path
        Path to the file.

    Returns
    -------
    DataFrame : DataFrame
        Pandas Dataframe with the data contained in the file.

    """
    h, data = read_qres_file(file_name)

    # Create header with channel names and units
    name_units = [n + " [" + u + "]" for n, u in zip(h["Description"], h["Units"])]

    # Create DataFrame
    df_data = pd.DataFrame(data=data, columns=name_units)

    # Set the time channel as index of the DataFrame
    df_data.set_index("Time [s]", inplace=True)

    return df_data


def check_file_type(file_name: str | Path):
    _file_ok = False

    cubus_id_string = "CaTs3 Time History File"
    simple_id_string = "CaTs3 Simple Time History Format 2011"

    with io.open(file_name, "rb") as f:
        #  line = f[0]
        n_bytes = len(cubus_id_string) * 2

        # Initialize name variable
        name = ""
        for _ in range(n_bytes):
            name += f.read(1).decode("utf-8").rstrip("\0")

        if name == cubus_id_string:
            _file_type = "CubusSimpleTimeHistory"
            _file_ok = True

        return _file_ok
