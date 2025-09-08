"""
References :
    https://www.pywhy.org/dowhy/v0.13/example_notebooks/gcm_online_shop.html#Step-3:-Answer-causal-questions
"""

from dowhy import gcm
from dowhy.utils import bar_plot
import pandas as pd
from typing import Tuple, Optional, Any, Dict
from importlib.resources import open_binary
import pickle
import numpy as np
import os

from agentic_supply.utilities.config import DATA_NAMES, DATA_TO_TARGET, ARTIFACTS_DIR
from agentic_supply import data
from agentic_supply.utilities.data_utils import get_data, save_object, visualise_graph
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
    >>> causal_graph = CausalGraph("example_data") # causal_graph = CausalGraph("online_shop_data")
    >>> causal_analysis = CausalAnalysis(causal_graph)
    >>> causal_analysis = CausalAnalysis(causal_graph, model_from_file=True)
    """

    def __init__(self, causal_graph: CausalGraph, model_from_file: bool = False):
        self.data_name: DATA_NAMES = causal_graph.data_name
        self.target = DATA_TO_TARGET[self.data_name]
        self.data = get_data(self.data_name)
        self.fit_report: Optional[str] = None
        self.evaluation_report: Optional[str] = None
        if model_from_file:
            self.model = self._load_model_from_file()
        else:
            self.model = gcm.InvertibleStructuralCausalModel(causal_graph.graph)  # StructuralCausalModel

        logger.info(f"Causal model instanciated for {self.data_name} with causal graph of form : {causal_graph.form}")

    def save_model(self):
        """
        Examples :
        >>> causal_analysis.save_model()
        >>> causal_analysis.fit().save_model()
        """
        save_object(self.model, filebasename=f"{self.data_name}_model")

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

    def get_intrinsic_causal_influence(self) -> Tuple[Dict, str]:
        """
        Examples :
        >>> ici, interpretation = causal_analysis.get_intrinsic_causal_influence()
        """
        ici = gcm.intrinsic_causal_influence(self.model, self.target)
        self._plot(
            basename="intrinsic_causal_influence",
            data=ici,
            ylabel="Variance attribution in %",
            title=f"Intrinsic causal influence plot for {self.data_name}",
        )
        interpretation = f"""The scores indicate how much variance each node is contributing to {self.target} — 
        without inheriting the variance from its parents in the causal graph (hence, intrinsic). 
        """
        return ici, interpretation

    def get_arrow_strength(self):
        """
        Examples :
        >>> as = causal_analysis.get_arrow_strength()
        """
        return gcm.arrow_strength(self.model, self.target)

    def get_anomaly_attribution(self, anomalous_data: pd.DataFrame, bootstrap: bool = False) -> Tuple[Dict, str]:
        """
        Examples :
        >>> anomalous_data = causal_analysis.generate_data(1)
        >>> anomalous_data["Y"] = 2 * anomalous_data["X"] + 15 # Here, we set the noise of Y to 15, which is unusually high.
        >>> anomalous_data["Z"] = 3 * anomalous_data["Y"]
        >>> aa, interpretation = causal_analysis.get_anomaly_attribution(anomalous_data)
        >>> aa, interpretation = causal_analysis.get_anomaly_attribution(anomalous_data, bootstrap=True)
        >>> anomalous_data.to_csv("./src/agentic_supply/data/anomalous_example_data.csv", index=False)
        """
        confidence_intervals = None
        if bootstrap:
            (node_contributions, confidence_intervals) = gcm.confidence_intervals(
                gcm.fit_and_compute(
                    gcm.attribute_anomalies,
                    self.model,
                    bootstrap_training_data=self.data,
                    target_node=self.target,
                    anomaly_samples=anomalous_data,
                ),
                num_bootstrap_resamples=10,
            )
        else:
            node_contributions = gcm.attribute_anomalies(self.model, self.target, anomaly_samples=anomalous_data)
            node_contributions = {k: v[0] for k, v in node_contributions.items()}
        most_impactful_node = self._get_most_impactful_node(node_contributions)
        interpretation = f"The node {most_impactful_node} has the highest likelihood of causing the anomaly."
        self._plot(
            basename="anomaly_attribution",
            data=node_contributions,
            uncertainties=confidence_intervals,
            ylabel="Anomaly attribution score",
            title=f"Anomaly attribution plot for {self.data_name}",
        )
        return node_contributions, interpretation

    def get_distribution_change_attribution(self, data_new: pd.DataFrame) -> Tuple[Dict, str]:
        """
        Examples :
        >>> import numpy as np
        >>> data_new = causal_analysis.generate_data(1000)
        >>> data_new["Y"] = 6 * data_new["X"] + np.random.normal(loc=0, scale=1, size=1000) # here, we set a different relation between X and Y
        >>> data_new["Z"] = 3 * data_new["Y"] + np.random.normal(loc=0, scale=1, size=1000)
        >>> dca, interpretation = causal_analysis.get_distribution_change_attribution(data_new)
        >>> data_new.to_csv("./src/agentic_supply/data/distribution_change_example_data.csv", index=False)
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

    def _convert_to_percentage(value_dictionary: dict):
        total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
        return {k: abs(v) / total_absolute_sum * 100 for k, v in value_dictionary.items()}

    def _load_model_from_file(self) -> gcm.InvertibleStructuralCausalModel:
        with open_binary(data, f"{self.data_name}_model.pkl") as f:
            return pickle.load(f)

    def _plot(self, basename: str, data: Dict, ylabel: str, title: str, in_memory: bool = False, uncertainties: Optional[Dict] = None):
        image_basename = f"{basename}_{self.data_name}"
        bar_plot(
            data,
            uncertainties=uncertainties,
            ylabel=ylabel,
            filename=os.path.join(ARTIFACTS_DIR, image_basename + ".png"),
            display_plot=False,
        )
        visualise_graph(image_basename, title, in_memory=in_memory)
