import networkx as nx


def generate_causal_graph():
    unit_price_causes = [
        (elem, "Unit Price") for elem in ("Country", "Shipment Mode", "Product Group", "Brand", "First Line Designation", "Managed by")
    ]
    causal_graph = nx.DiGraph([("X", "Y"), ("Y", "Z")])
