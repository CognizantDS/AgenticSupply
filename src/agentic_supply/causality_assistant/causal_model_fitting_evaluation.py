from dowhy import gcm
import pandas as pd
import networkx as nx
import uuid
import os
import matplotlib.pyplot as plt
from typing import Tuple

from agentic_supply.utilities.config import DATA_NAMES, ARTIFACTS_DIR
from agentic_supply.utilities.data_utils import get_data, write_png_to_html


def fit_eval_causal_model(
    data_name: DATA_NAMES, causal_graph: nx.DiGraph
) -> Tuple[gcm.StructuralCausalModel, gcm.model_evaluation.CausalModelEvaluationResult, str]:
    data = get_data(data_name)
    causal_model = gcm.StructuralCausalModel(causal_graph)
    # gcm.auto.assign_causal_mechanisms(model, data)
    gcm.fit(causal_model, data)
    evaluation = gcm.evaluate_causal_model(causal_model, data)
    image_basename = f"causal_model_evaluation_{uuid.uuid4().hex}"
    image_filepath_png, image_filepath_html = (os.path.join(ARTIFACTS_DIR, image_basename + extension) for extension in [".png", ".html"])
    plt.savefig(image_filepath_png)
    plt.clf()
    write_png_to_html(png_path=image_filepath_png, html_path=image_filepath_html, title=f"Causal Model Evaluation for {data_name}")
    return causal_model, evaluation, image_filepath_html
