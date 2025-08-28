import os
import pandas as pd
import base64
from importlib.resources import files, as_file
from typing import Optional

from agentic_supply import data
from agentic_supply.utilities.config import DATA_NAMES, DATA_TO_FILE


def get_data(data_name: DATA_NAMES) -> pd.DataFrame:
    source = files(data).joinpath(DATA_TO_FILE[data_name])
    with as_file(source) as myfile:
        df = pd.read_csv(myfile)
    return df


def write_png_to_html(png_path: str, title: str, html_path: Optional[str] = None):
    with open(png_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    # encoded_string = f"data:image/png;base64,{encoded}"
    # encoded_html = f"<img src='data:image/png;base64,{encoded}' alt='Causal graph image' />"
    # markdown_command = f"![Causal graph image](http://localhost:8765/{image_filepath_png})"

    encoded_html = f"""
    <html>
    <body>
        <h2>{title}</h2>
        <img src='data:image/png;base64,{encoded}' alt={title}' />
    </body>
    </html>
    """

    if html_path is None:
        html_path = os.path.splitext(png_path)[0] + ".html"

    with open(html_path, "w") as f:
        f.write(encoded_html)
