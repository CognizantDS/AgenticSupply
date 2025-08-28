"""
References :
    https://github.com/cognizant-ai-lab/neuro-san/blob/main/neuro_san/coded_tools/math_guy/calculator.py
    https://github.com/cognizant-ai-lab/neuro-san-studio/blob/main/docs/tutorial.md#custom-tools
    https://github.com/cognizant-ai-lab/neuro-san/blob/main/docs/agent_hocon_reference.md#class
"""

import os
import networkx as nx  # from dowhy.utils import plot
import matplotlib.pyplot as plt
import uuid
import base64
import pickle
from typing import Dict, List, Tuple

from agentic_supply.utilities.config import DATA_NAMES, ARTIFACTS_DIR
from agentic_supply.utilities.data_utils import write_png_to_html

DATA_TO_GRAPH_FORM: Dict[DATA_NAMES, List[Tuple]] = {
    "example_data": [("X", "Y"), ("Y", "Z")],
    "supply_chain_medical": [
        ("Country", "Managed By"),
        ("Managed By", "Fulfill Via"),
        ("Managed By", "Shipment Mode"),
        ("Fulfill Via", "Shipment Mode"),
        ("Country", "Shipment Mode"),
        ("Country", "Product Group"),
        ("Brand", "Product Group"),
        ("Product Group", "Sub Classification"),
        ("Sub Classification", "Molecule/Test Type"),
        # Price
        ("Shipment Mode", "Freight Cost (USD)"),
        ("Freight Cost (USD)", "Unit Price"),
        ("Sub Classification", "Unit Price"),
        ("Molecule/Test Type", "Unit Price"),
    ],
    "supply_chain_logistics": [("demand", "submitted"), ("constraint", "submitted"), ("submitted", "confirmed"), ("confirmed", "received")],
}


def generate_causal_graph(data_name: DATA_NAMES) -> Tuple[str, str]:
    causal_graph = nx.DiGraph(DATA_TO_GRAPH_FORM[data_name])
    image_basename = f"causal_graph_{uuid.uuid4().hex}"
    image_filepath_png, image_filepath_html, causal_graph_path = (
        os.path.join(ARTIFACTS_DIR, image_basename + extension) for extension in [".png", ".html", ".pkl"]
    )
    with open(causal_graph_path, "wb") as obj_file:
        pickle.dump(causal_graph, obj_file)
    nx.draw_networkx(causal_graph)  # plot(causal_graph)
    plt.savefig(image_filepath_png)
    plt.clf()
    write_png_to_html(png_path=image_filepath_png, html_path=image_filepath_html, title=f"Causal Graph for {data_name}")

    return (causal_graph_path, image_filepath_html)
