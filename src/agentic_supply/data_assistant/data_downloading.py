"""
References :
"""

import os
import tkinter as tk
from tkinter import filedialog
import shutil
from importlib.resources import files, as_file
from typing import Dict, List, Tuple

from agentic_supply.utilities.log_utils import set_logging, get_logger
from agentic_supply.utilities.config import DATA_NAMES, DATA_TO_FILE
from agentic_supply import data

set_logging()
logger = get_logger(__name__)


def download_data(data_name: DATA_NAMES, open_file: bool = True) -> str:
    source = files(data).joinpath(data_name)
    with as_file(source) as myfile:
        root = tk.Tk()
        root.withdraw()
        target_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")], title=f"Save {data_name} As"
        )
        target_path = os.path.abspath(target_path)

        if target_path:
            logger.info(f"Copying file from {myfile} to {target_path}")
            shutil.copyfile(myfile, target_path)

    if open_file:
        os.startfile(target_path)

    return target_path
