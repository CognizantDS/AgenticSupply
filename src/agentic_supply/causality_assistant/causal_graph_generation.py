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
import webbrowser
from typing import Dict, List, Tuple, Optional

from agentic_supply.utilities.config import DATA_NAMES, ARTIFACTS_DIR
from agentic_supply.utilities.data_utils import write_png_to_html
from agentic_supply.utilities.log_utils import set_logging, get_logger


set_logging()
logger = get_logger(__name__)


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
    "supply_chain_logistics": [
        ("demand", "submitted"),
        ("constraint", "submitted"),
        ("submitted", "confirmed"),
        ("confirmed", "received"),
    ],
}


class CausalGraph:
    def __init__(self, data_name: Optional[DATA_NAMES] = None, form: Optional[List[Tuple]] = None):
        self.data_name: DATA_NAMES = data_name
        self.form: List[Tuple] = DATA_TO_GRAPH_FORM[data_name] if data_name is not None and form is None else form
        self.graph: Optional[nx.DiGraph] = None
        logger.info(f"Causal graph instanciated with form : {self.form}")

    def generate(self) -> "CausalGraph":
        self.graph = nx.DiGraph(self.form)
        return self

    def visualise(self) -> Tuple[str, str]:
        image_basename = f"causal_graph_{uuid.uuid4().hex}"
        image_filepath_png, image_filepath_html, causal_graph_path = (
            os.path.join(ARTIFACTS_DIR, image_basename + extension) for extension in [".png", ".html", ".pkl"]
        )
        with open(causal_graph_path, "wb") as obj_file:
            pickle.dump(self.graph, obj_file)
        nx.draw_networkx(self.graph)  # plot(causal_graph)
        plt.savefig(image_filepath_png)
        plt.clf()
        write_png_to_html(png_path=image_filepath_png, html_path=image_filepath_html, title=f"Causal Graph for {self.data_name}")
        webbrowser.open_new_tab(f"file://{os.path.abspath(image_filepath_html)}")
        return (causal_graph_path, image_filepath_html)
