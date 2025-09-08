import os
import pickle
import pandas as pd
import base64
import webbrowser
import matplotlib.pyplot as plt
from importlib.resources import files, as_file
from typing import Optional


from agentic_supply import data
from agentic_supply.utilities.config import DATA_NAMES, DATA_TO_FILE, ARTIFACTS_DIR
from agentic_supply.utilities.log_utils import set_logging, get_logger


set_logging()
logger = get_logger(__name__)


def save_object(object_: object, filebasename):
    source = files(data).joinpath(f"{filebasename}.pkl")
    with open(source, "wb") as out:
        pickle.dump(object_, out, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved object at {source}")


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


def visualise_graph(image_basename: str, title: str, in_memory: bool = True):
    image_filepath_png, image_filepath_html = (os.path.join(ARTIFACTS_DIR, image_basename + extension) for extension in [".png", ".html"])
    if in_memory:
        plt.savefig(image_filepath_png)
        plt.clf()
    write_png_to_html(png_path=image_filepath_png, html_path=image_filepath_html, title=title)
    webbrowser.open_new_tab(f"file://{os.path.abspath(image_filepath_html)}")
