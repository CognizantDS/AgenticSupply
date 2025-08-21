"""
References :
    https://github.com/cognizant-ai-lab/neuro-san/blob/main/neuro_san/coded_tools/math_guy/calculator.py
    https://github.com/cognizant-ai-lab/neuro-san-studio/blob/main/docs/tutorial.md#custom-tools
    https://github.com/cognizant-ai-lab/neuro-san/blob/main/docs/agent_hocon_reference.md#class
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import uuid
import base64


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
    image_dir = "./logs"
    image_basename = f"causal_graph_{uuid.uuid4().hex}"
    image_filepath_png, image_filepath_html = (os.path.join(image_dir, image_basename + extension) for extension in [".png", ".html"])
    nx.draw_networkx(causal_graph)
    plt.savefig(image_filepath_png)

    with open(image_filepath_png, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    # encoded_string = f"data:image/png;base64,{encoded}"
    # encoded_html = f"<img src='data:image/png;base64,{encoded}' alt='Causal graph image' />"
    # markdown_command = f"![Causal graph image](http://localhost:8765/{image_filepath_png})"

    encoded_html = f"""
    <html>
    <body>
        <h2>Causal Graph</h2>
        <img src='data:image/png;base64,{encoded}' alt='Causal graph image' />
    </body>
    </html>
    """

    with open(image_filepath_html, "w") as f:
        f.write(encoded_html)

    return (causal_graph, image_filepath_html)
