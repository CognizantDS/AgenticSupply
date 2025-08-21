from dowhy import gcm
import pandas as pd
import networkx as nx


def fit_causal_model(data: pd.DataFrame, graph: nx.DiGraph) -> gcm.StructuralCausalModel:
    # graph_str = "\n".join(nx.generate_gml(graph))
    model = gcm.StructuralCausalModel(graph)
    gcm.auto.assign_causal_mechanisms(model, data)
    gcm.fit(model, data)
    return model
