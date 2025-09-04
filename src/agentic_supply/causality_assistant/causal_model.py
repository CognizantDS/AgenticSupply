from dowhy import gcm
import pandas as pd
import networkx as nx
import uuid
import os
import matplotlib.pyplot as plt
from typing import Tuple

from agentic_supply.utilities.config import DATA_NAMES, ARTIFACTS_DIR
from agentic_supply.utilities.data_utils import get_data, write_png_to_html


class CausalModel:
    def __init__(self, data_name: DATA_NAMES, causal_graph: nx.DiGraph):
        self.causal_model = gcm.StructuralCausalModel(causal_graph)
        self.data = get_data(data_name)
        self.data_name: DATA_NAMES = data_name

    def fit(self):
        gcm.auto.assign_causal_mechanisms(self.causal_model, self.data)
        gcm.fit(self.causal_model, self.data)

    def evaluate(self):
        evaluation = gcm.evaluate_causal_model(self.causal_model, self.data)
        image_basename = f"causal_model_evaluation_{uuid.uuid4().hex}"
        image_filepath_png, image_filepath_html = (
            os.path.join(ARTIFACTS_DIR, image_basename + extension) for extension in [".png", ".html"]
        )
        plt.savefig(image_filepath_png)
        plt.clf()
        write_png_to_html(png_path=image_filepath_png, html_path=image_filepath_html, title=f"Causal Model Evaluation for {self.data_name}")
        return evaluation, image_filepath_html
