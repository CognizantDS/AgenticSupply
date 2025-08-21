"""
References :
    https://github.com/cognizant-ai-lab/neuro-san/blob/main/neuro_san/coded_tools/math_guy/calculator.py
    https://github.com/cognizant-ai-lab/neuro-san-studio/blob/main/docs/tutorial.md#custom-tools
    https://github.com/cognizant-ai-lab/neuro-san/blob/main/docs/agent_hocon_reference.md#class
"""

import networkx as nx
import matplotlib.pyplot as plt
import uuid


def generate_causal_graph() -> nx.DiGraph:
    causal_graph = nx.DiGraph(
        [
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
        ]
    )
    image_path = f"logs/causal_graph_{uuid.uuid4().hex}.png"
    nx.draw_networkx(causal_graph)
    plt.savefig(image_path)
    markdown_command = f"![Causal graph image]({image_path})"

    return (causal_graph, markdown_command)
