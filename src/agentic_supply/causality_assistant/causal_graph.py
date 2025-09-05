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
import webbrowser
from typing import Dict, List, Tuple, Optional
from dowhy.gcm.falsify import falsify_graph, EvaluationResult

from agentic_supply.utilities.config import DATA_NAMES, ARTIFACTS_DIR
from agentic_supply.utilities.data_utils import write_png_to_html, get_data
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
    """
    Causal graph for supported data

    Examples :
    >>> from agentic_supply.causality_assistant.causal_graph import CausalGraph
    >>> causal_graph = CausalGraph("example_data")
    """

    def __init__(self, data_name: DATA_NAMES, form: Optional[List[Tuple]] = None):
        self.data_name: DATA_NAMES = data_name
        self.form: List[Tuple] = form if form is not None else DATA_TO_GRAPH_FORM[data_name]
        self.graph = nx.DiGraph(self.form)
        self.id: str = uuid.uuid4().hex
        self.refutation: Optional[EvaluationResult] = None
        self.refutation_report: Optional[str] = None
        logger.info(f"Causal graph instanciated for {self.data_name} with form : {self.form}")

    def visualise(self) -> "CausalGraph":
        """
        Examples :
        >>> causal_graph.visualise()
        """
        image_basename = f"causal_graph_{self.id}"
        nx.draw_networkx(self.graph)
        visualise_graph(image_basename, f"Causal Graph for {self.data_name}")
        return self

    def refutate(self) -> "CausalGraph":
        """
        Examples :
        >>> causal_graph.refutate()
        """
        image_basename = f"causal_graph_refutation_{self.id}"
        png_path = os.path.join(ARTIFACTS_DIR, image_basename + ".png")
        data = get_data(self.data_name)
        self.refutation = falsify_graph(
            self.graph,
            data,
            show_progress_bar=True,
            plot_histogram=True,
            plot_kwargs={"savepath": png_path, "display": False},
        )
        visualise_graph(image_basename, f"Causal Graph refutation report for {self.data_name}", in_memory=False)
        self.refutation_report = f"Graph is falsifiable: {self.refutation.falsifiable}, Graph is falsified: {self.refutation.falsified}\n\n{repr(self.refutation)}"
        return self


def visualise_graph(image_basename: str, title=str, in_memory: bool = True):
    image_filepath_png, image_filepath_html = (os.path.join(ARTIFACTS_DIR, image_basename + extension) for extension in [".png", ".html"])
    if in_memory:
        plt.savefig(image_filepath_png)
        plt.clf()
    write_png_to_html(png_path=image_filepath_png, html_path=image_filepath_html, title=title)
    webbrowser.open_new_tab(f"file://{os.path.abspath(image_filepath_html)}")
