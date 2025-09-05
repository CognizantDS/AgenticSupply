from dowhy import gcm
import pandas as pd
from typing import Tuple, Optional, Any, Dict
from importlib.resources import open_binary
import pickle
import numpy as np

from agentic_supply.utilities.config import DATA_NAMES, DATA_TO_TARGET
from agentic_supply import data
from agentic_supply.utilities.data_utils import get_data
from agentic_supply.causality_assistant.causal_graph import CausalGraph
from agentic_supply.utilities.log_utils import set_logging, get_logger


set_logging()
logger = get_logger(__name__)


class CausalAnalysis:
    """
    Causal model for supported data

    Examples :
    >>> from agentic_supply.causality_assistant.causal_analysis import CausalAnalysis
    >>> from agentic_supply.causality_assistant.causal_graph import CausalGraph
    >>> causal_graph = CausalGraph("example_data")
    >>> causal_analysis = CausalAnalysis(causal_graph)
    """

    def __init__(self, causal_graph: CausalGraph):
        self.model = gcm.InvertibleStructuralCausalModel(causal_graph.graph)  # StructuralCausalModel
        self.data_name: DATA_NAMES = causal_graph.data_name
        self.target = DATA_TO_TARGET[self.data_name]
        self.data = get_data(self.data_name)
        self.fit_report: Optional[str] = None
        self.evaluation_report: Optional[str] = None
        logger.info(f"Causal model instanciated for {self.data_name} with causal graph of form : {causal_graph.form}")

    @staticmethod
    def from_file(data_name: DATA_NAMES) -> "CausalAnalysis":
        """
        Examples :
        >>> causal_analysis = CausalAnalysis.from_file("example_data")
        """
        with open_binary(data, f"{data_name}_model.pkl") as f:
            return pickle.load(f)

    def fit(self) -> "CausalAnalysis":
        """
        Examples :
        >>> causal_analysis.fit()
        >>> print(causal_analysis.fit_report)
        """
        summary_auto_assignment = gcm.auto.assign_causal_mechanisms(self.model, self.data)
        gcm.fit(self.model, self.data)
        self.fit_report = str(summary_auto_assignment)
        return self

    def evaluate(self) -> "CausalAnalysis":
        """
        Examples :
        >>> causal_analysis.evaluate()
        >>> print(causal_analysis.evaluation_report)
        """
        evaluation = gcm.evaluate_causal_model(self.model, self.data, evaluate_causal_structure=False)
        self.evaluation_report = str(evaluation)
        return self

    def generate_data(self, num_samples=100) -> pd.DataFrame:
        """
        Examples :
        >>> data = causal_analysis.generate_data()
        """
        return gcm.draw_samples(self.model, num_samples)

    def generate_interventional_samples(self, node: str, num_samples=100, intervention_function=lambda x: x + 0.5) -> pd.DataFrame:
        """
        Examples :
        >>> data = causal_analysis.generate_interventional_samples("X")
        """
        return gcm.interventional_samples(self.model, {node: intervention_function}, num_samples_to_draw=num_samples)

    def generate_counterfactual_samples(
        self,
        node: str,
        intervention_function=lambda x: x + 0.5,
        observed_data: Optional[pd.DataFrame] = None,
        noise_data: Optional[pd.DataFrame] = None,
    ):
        """
        Examples :
        >>> data = causal_analysis.generate_counterfactual_samples("X")
        """
        return gcm.counterfactual_samples(self.model, {node: intervention_function}, observed_data=observed_data, noise_data=noise_data)

    def get_average_causal_effect(self) -> float:
        """
        Examples :
        >>> ace = causal_analysis.get_average_causal_effect()
        """
        return gcm.average_causal_effect(
            self.model,
            self.target,
            interventions_alternative={"X": lambda x: 1},
            interventions_reference={"X": lambda x: 0},
            observed_data=self.data,
        )

    def get_intrinsic_causal_influence(self):
        """
        Examples :
        >>> ici = causal_analysis.get_intrinsic_causal_influence()
        """
        return gcm.intrinsic_causal_influence(self.model, self.target)

    def get_arrow_strength(self):
        """
        Examples :
        >>> as = causal_analysis.get_arrow_strength()
        """
        return gcm.arrow_strength(self.model, self.target)

    def get_anomaly_attribution(self, anomalous_data: pd.DataFrame) -> Tuple[Dict, str]:
        """
        Examples :
        >>> anomalous_data = causal_analysis.generate_data(1)
        >>> anomalous_data["Y"] = 2 * anomalous_data["X"] + 5 # Here, we set the noise of Y to 5, which is unusually high.
        >>> anomalous_data["Z"] = 3 * anomalous_data["Y"]
        >>> aa, interpretation = causal_analysis.get_anomaly_attribution(anomalous_data)
        """
        node_contributions = gcm.attribute_anomalies(self.model, self.target, anomaly_samples=anomalous_data)
        most_impactful_node = self._get_most_impactful_node(node_contributions)
        interpretation = f"The node {most_impactful_node} has the highest likelihood of causing the anomaly."
        return node_contributions, interpretation

    def get_distribution_change_attribution(self, data_new: pd.DataFrame) -> Tuple[Dict, str]:
        """
        Examples :
        >>> data_new = causal_analysis.generate_data(1000)
        >>> data_new["Y"] = 6 * data_new["X"] + np.random.normal(loc=0, scale=1, size=1000)
        >>> data_new["Z"] = 3 * data_new["Y"] + np.random.normal(loc=0, scale=1, size=1000)
        >>> dca, interpretation = causal_analysis.get_distribution_change_attribution(data_new)
        """
        node_attributions = gcm.distribution_change(self.model, self.data, data_new, self.target)
        most_impactful_node = self._get_most_impactful_node(node_attributions)
        interpretation = f"The node {most_impactful_node} has the highest likelihood of causing the distribution change."
        return node_attributions, interpretation

    def get_feature_relevance(self) -> Tuple[Dict, np.ndarray, str]:
        """
        Examples :
        >>> parent_relevance, noise_relevance, interpretation = causal_analysis.get_feature_relevance()
        """
        parent_relevance, noise_relevance = gcm.parent_relevance(self.model, target_node=self.target)
        most_impactful_node = self._get_most_impactful_node(parent_relevance)
        interpretation = f"The relation {most_impactful_node} has the highest relevance to {self.target}."
        return parent_relevance, noise_relevance, interpretation

    def _get_most_impactful_node(self, impact: dict) -> str:
        avg_impact = {k: np.mean(v) for k, v in impact.items()}
        return max(avg_impact, key=avg_impact.get)
