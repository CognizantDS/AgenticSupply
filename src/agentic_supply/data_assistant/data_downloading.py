"""
References :
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import shutil
from importlib.resources import files, as_file
from typing import Dict, List, Tuple, Optional, Literal

from agentic_supply.utilities.log_utils import set_logging, get_logger
from agentic_supply.utilities.config import DATA_NAMES, DATA_TO_FILE
from agentic_supply import data

set_logging()
logger = get_logger(__name__)


def select_target_path(mode: Literal["save", "openname"]) -> str:
    root = tk.Tk()
    root.withdraw()
    if mode == "openname":
        target_path = filedialog.askopenfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Use")
    elif mode == "save":
        target_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title="Save as")
    target_path = os.path.abspath(target_path)
    return target_path


def download_data(df: Optional[pd.DataFrame] = None, data_name: Optional[DATA_NAMES] = None, open_file: bool = True) -> str:
    """
    Examples :
    >>> target_path = download_data(data_name="example_data")
    >>> target_path = download_data(df=pd.DataFrame({"X": [1, 2], "Y": [3, 4]}))
    """
    target_path = select_target_path("save")

    if data_name is not None:
        source = files(data).joinpath(f"{data_name}.csv")
        with as_file(source) as myfile:
            if target_path:
                logger.info(f"Copying file from {myfile} to {target_path}")
                shutil.copyfile(myfile, target_path)
    elif df is not None:
        logger.info(f"Saving dataframe to {target_path}")
        df.to_csv(target_path, index=False)
    else:
        raise ValueError("Either 'data_name' or 'df' must be provided to save data !")

    if open_file:
        os.startfile(target_path)

    return target_path
