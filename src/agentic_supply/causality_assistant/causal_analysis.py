"""
The analysis process is :
- 1) Build the causal graph based on domain knowledge / refute to check
    https://www.pywhy.org/dowhy/v0.13/user_guide/modeling_causal_relations/specifying_causal_graph.html
    https://www.pywhy.org/dowhy/v0.13/user_guide/modeling_causal_relations/refuting_causal_graph/refute_causal_structure.html
- 2) Automatically infer causal mechanisms and fit an InvertibleStructuralCausalModel / evaluate
    https://www.pywhy.org/dowhy/v0.13/user_guide/modeling_gcm/model_evaluation.html
- 3) Perform causal tasks
    A) Estimating causal effects
        https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/estimating_causal_effects/effect_estimation_with_gcm.html
    B) Quantify causal influence
        https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/quantify_causal_influence/quantify_arrow_strength.html
        https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/quantify_causal_influence/icc.html
    C) Root cause analysis
        https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/root_causing_and_explaining/anomaly_attribution.html
        https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/root_causing_and_explaining/distribution_change.html
        https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/root_causing_and_explaining/feature_relevance.html
    D) Asking and answering what-if questions
        https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/what_if/interventions.html
        https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/what_if/counterfactuals.html
    E) Predicting outcome for out-of-distribution inputs
        https://www.pywhy.org/dowhy/v0.13/user_guide/causal_tasks/causal_prediction/index.html

References :
    https://www.pywhy.org/dowhy/v0.13/example_notebooks/gcm_online_shop.html#Step-3:-Answer-causal-questions
"""

from dowhy import gcm  # this takes a while, but used everywhere
import pandas as pd
from typing import Tuple, Optional, Any, Dict
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
    Causal analysis for supported data

    Examples :
    >>> from agentic_supply.causality_assistant.causal_analysis import CausalAnalysis
    >>> from agentic_supply.causality_assistant.causal_graph import CausalGraph
    >>> data_name = "example_data" # "online_shop_data" "supply_chain_logistics"
    >>> causal_graph = CausalGraph(data_name)
    >>> causal_analysis = CausalAnalysis(causal_graph)
    >>> causal_analysis = CausalAnalysis(causal_graph, model_from_file=True)
    """

    def __init__(self, causal_graph: CausalGraph, model_from_file: bool = False):
        self.causal_graph: CausalGraph = causal_graph
        self.data_name: DATA_NAMES = causal_graph.data_name
        self.target = DATA_TO_TARGET[self.data_name]
        self.data = get_data(self.data_name)
        self.fit_report: Optional[str] = None
        self.evaluation_report: Optional[str] = None
        if model_from_file:
            self.model = self._load_model_from_file()
        else:
            self.model = gcm.InvertibleStructuralCausalModel(self.causal_graph.graph)  # StructuralCausalModel
        logger.info(f"Causal model instanciated for {self.data_name} with causal graph of form : {self.causal_graph.form}")

    def save_model(self):
        """
        Examples :
        >>> causal_analysis.save_model()
        >>> causal_analysis.fit().save_model()
        """
        save_object(self.model, filebasename=f"{self.data_name}_model")

    @staticmethod
    def _convert_to_percentage(value_dictionary: dict) -> Dict:
        total_absolute_sum = np.sum([abs(v) for v in value_dictionary.values()])
        return {k: float(abs(v) / total_absolute_sum * 100) for k, v in value_dictionary.items()}

    @staticmethod
    def _str_to_lambda(expression_str: str) -> Any:
        """
        This enables to have lambda expressions from a string input, for the user to give an intervention function as a string.
        Examples :
        >>> lambda_fn = causal_analysis._str_to_lambda("x + 5") # shift
        >>> lambda_fn = causal_analysis._str_to_lambda("x * 5") # proportional
        >>> lambda_fn = causal_analysis._str_to_lambda("5") # atomic
        """
        import sympy as sp
        import dis
        from io import StringIO

        parsed_expr = sp.sympify(expression_str)
        lambda_fn = sp.lambdify(sp.symbols("x"), parsed_expr)
        with StringIO() as out:
            dis.dis(lambda_fn, file=out)
            logger.info(f"Parsed {expression_str} to :\n{out.getvalue()}")
        return lambda_fn

    @staticmethod
    def _get_most_impactful_node(impact: dict) -> str:
        avg_impact = {k: np.mean(v) for k, v in impact.items()}
        return max(avg_impact, key=avg_impact.get)

    # Model fitting and evaluation
    def fit(self) -> "CausalAnalysis":
        """
        Examples :
        >>> causal_analysis.fit()
        >>> print(causal_analysis.fit_report)
        """
        logger.info(f"Fitting the model for {self.data_name} with causal graph of form : {self.causal_graph.form}")
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
        logger.info(f"Evaluating the fitted model for {self.data_name} with causal graph of form : {self.causal_graph.form}")
        evaluation = gcm.evaluate_causal_model(self.model, self.data, evaluate_causal_structure=False)
        self.evaluation_report = str(evaluation)
        return self

    # Causal tasks
    ## Asking and answering What-If questions
    def generate_data(self, num_samples=100) -> pd.DataFrame:
        """
        Examples :
        >>> data = causal_analysis.generate_data()
        """
        logger.info(f"Generating data for {num_samples} samples")
        return gcm.draw_samples(self.model, num_samples)

    def generate_interventional_samples(
        self,
        node: str,
        intervention_str: str = "x + 0.5",
        num_samples: int = 5,
        observed_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Question : What will happen to the variable Z if I intervene on Y ? (= future)
        Either pass num_samples to draw from generated data, or pass observed data.
        Examples :
        >>> data = causal_analysis.generate_interventional_samples("X", num_samples=10)
        >>> data = causal_analysis.generate_interventional_samples("X", intervention_str="x * 0.5") # soft intervention
        >>> data = causal_analysis.generate_interventional_samples("X", intervention_str="x + 5") # shift intervention
        >>> data = causal_analysis.generate_interventional_samples("X", intervention_str="5") # atomic intervention
        >>> data = causal_analysis.generate_interventional_samples("X", observed_data=causal_analysis.data)
        >>> data = causal_analysis.generate_interventional_samples("X", observed_data=causal_analysis.data, intervention_str="5")
        """
        if observed_data is not None:
            num_samples = None
        logger.info(f"Generating {num_samples} interventional samples, with {intervention_str} intervention on node {node}")
        return gcm.interventional_samples(
            self.model, {node: self._str_to_lambda(intervention_str)}, num_samples_to_draw=num_samples, observed_data=observed_data
        )

    def generate_counterfactual_samples(
        self,
        node: str,
        intervention_str: str = "x + 0.5",
        observed_data: Optional[pd.DataFrame] = None,
        noise_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Question : I observed a certain outcome z for a variable Z where variable X was set to a value x :
        what would have happened to the value of Z, had I intervened on X to assign it a different value x' ? (= alternative past)
        Either pass observed_data to generate counterfactuals from, or pass noise_data.
        Examples :
        >>> data = causal_analysis.generate_counterfactual_samples("X", observed_data=causal_analysis.data.iloc[[0]], intervention_str="5")
        >>> causal_analysis.data.iloc[[0]].to_csv("./src/agentic_supply/data/counterfactual_example_data.csv", index=False)
        """
        num_samples = len(observed_data) if observed_data is not None else len(noise_data)
        logger.info(f"Generating {num_samples} counterfactual samples, with '{intervention_str}' intervention on node {node}")
        return gcm.counterfactual_samples(
            self.model, {node: self._str_to_lambda(intervention_str)}, observed_data=observed_data, noise_data=noise_data
        )

    ## Estimating causal effects
    def get_average_causal_effect(
        self, interventions_alternative: Dict = {"X": "1"}, interventions_reference: Dict = {"X": "0"}
    ) -> Tuple[float, str]:
        """
        Question : How much does a certain target quantity differ under two different interventions/treatments ?
        Examples :
        >>> ace, interpretation = causal_analysis.get_average_causal_effect()
        >>> ace, interpretation = causal_analysis.get_average_causal_effect(interventions_alternative = {"X": "0"}, interventions_reference = {"X": "1"})
        """
        logger.info(
            f"Calculating average causal effect from the difference between alternative '{interventions_alternative}' and reference '{interventions_reference}'"
        )
        ace = gcm.average_causal_effect(
            self.model,
            self.target,
            interventions_alternative={k: self._str_to_lambda(v) for k, v in interventions_alternative.items()},
            interventions_reference={k: self._str_to_lambda(v) for k, v in interventions_reference.items()},
            observed_data=self.data,
        )
        interpretation = f"""The target quantity of {self.target} differs on average by {ace} units,
        between interventions_reference {interventions_reference} and interventions_alternative {interventions_alternative}. 
        """
        return ace, interpretation

    ## Quantify causal influence
    def get_arrow_strength(self) -> Tuple[Dict, Dict, str]:
        """
        Question : How strong is the causal influence from a cause to its direct effect ?
        Examples :
        >>> node_contributions, node_contributions_pct, interpretation = causal_analysis.get_arrow_strength()
        """
        logger.info(f"Calculating arrow strength of parent nodes to {self.target}")
        node_contributions = gcm.arrow_strength(self.model, self.target)
        node_contributions_pct = self._convert_to_percentage(node_contributions)
        most_impactful_node = self._get_most_impactful_node(node_contributions)
        interpretation = f"""Arrow strength (direct effect) scores : {node_contributions} (percentages : {node_contributions_pct}). 
        The scores indicate how much variance each node is contributing to {self.target} — where influences through paths over other nodes are ignored. 
        Removing the most impactful causal link, {most_impactful_node}, increases the variance of {self.target} by {node_contributions[most_impactful_node]} units ({node_contributions_pct[most_impactful_node]} %).
        """
        return node_contributions, node_contributions_pct, interpretation

    def get_intrinsic_causal_influence(self) -> Tuple[Dict, Dict, str]:
        """
        Question : How strong is the causal influence of an upstream node to a target node that is not inherited from the parents of the upstream node ?
        Examples :
        >>> node_contributions, node_contributions_pct, interpretation = causal_analysis.get_intrinsic_causal_influence()
        """
        logger.info(f"Calculating intrinsic causal influence of parent nodes to {self.target}")
        node_contributions = gcm.intrinsic_causal_influence(self.model, self.target)
        self._plot(
            basename="intrinsic_causal_influence",
            data=node_contributions,
            ylabel="Variance attribution in %",
            title=f"Intrinsic causal influence plot for {self.data_name}",
        )
        node_contributions = {k: float(v) for k, v in node_contributions.items()}
        node_contributions_pct = self._convert_to_percentage(node_contributions)
        most_impactful_node = self._get_most_impactful_node(node_contributions)
        interpretation = f"""Intrinsic causal influence scores : {node_contributions} (percentages : {node_contributions_pct}).
        The scores indicate how much variance each node is contributing to {self.target} — without inheriting the variance from its parents in the causal graph (hence, intrinsic to the node itself).
        The most impactful node {most_impactful_node} contributes {node_contributions_pct[most_impactful_node]} % of the variance in {self.target}.
        """
        return node_contributions, node_contributions_pct, interpretation

    ## Root cause analysis
    def get_anomaly_attribution(self, anomalous_data: pd.DataFrame, bootstrap: bool = False) -> Tuple[Dict, str]:
        """
        Question : How much did each of the upstream nodes and the target node contribute to the observed anomaly ?
        Examples :
        >>> anomalous_data = causal_analysis.generate_data(1)
        >>> anomalous_data["Y"] = 2 * anomalous_data["X"] + 15 # Here, we set the noise of Y to 15, which is unusually high.
        >>> anomalous_data["Z"] = 3 * anomalous_data["Y"]
        >>> aa, interpretation = causal_analysis.get_anomaly_attribution(anomalous_data)
        >>> aa, interpretation = causal_analysis.get_anomaly_attribution(anomalous_data, bootstrap=True)
        >>> anomalous_data.to_csv("./src/agentic_supply/data/anomalous_example_data.csv", index=False)
        """
        logger.info(
            f"Calculating anomaly attribution from anomalous_data with {len(anomalous_data)} samples {'using the bootstrap method' if bootstrap else ''}"
        )
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
        node_contributions = {k: float(v) for k, v in node_contributions.items()}
        self._plot(
            basename="anomaly_attribution",
            data=node_contributions,
            uncertainties=confidence_intervals,
            ylabel="Anomaly attribution score",
            title=f"Anomaly attribution plot for {self.data_name}",
        )
        most_impactful_node = self._get_most_impactful_node(node_contributions)
        interpretation = f"""Anomaly likelihood scores : {node_contributions}.
        The node {most_impactful_node} has the highest likelihood of causing the anomaly seen in your given data.
        A positive attribution score means that the corresponding node contributed to the observed anomaly, which is in our case the drop in Profit. 
        A negative score of a node indicates that the observed value for the node is actually reducing the likelihood of the anomaly"""
        return node_contributions, interpretation

    def get_distribution_change_attribution(
        self, data_new: pd.DataFrame, data_old: Optional[pd.DataFrame] = None, bootstrap: bool = False
    ) -> Tuple[Dict, str]:
        """
        Question : What mechanism in my system changed between two sets of data ? Or in other words, which node in my data behaves differently ?
        Examples :
        >>> import numpy as np
        >>> data_new = causal_analysis.generate_data(100)
        >>> data_new["Y"] = 6 * data_new["X"] + np.random.normal(loc=0, scale=1, size=100) # here, we set a different relation between X and Y
        >>> data_new["Z"] = 3 * data_new["Y"] + np.random.normal(loc=0, scale=1, size=100)
        >>> dca, interpretation = causal_analysis.get_distribution_change_attribution(data_new)
        >>> dca, interpretation = causal_analysis.get_distribution_change_attribution(data_new, bootstrap=True)
        >>> data_new.to_csv("./src/agentic_supply/data/distribution_change_example_data.csv", index=False)
        """
        logger.info(
            f"Calculating distribution change attribution from data_new with {len(data_new)} samples {'using the bootstrap method' if bootstrap else ''}"
        )
        confidence_intervals = None
        if data_old is None:
            data_old = self.data
        if bootstrap:
            node_contributions, confidence_intervals = gcm.confidence_intervals(
                gcm.bootstrap_sampling(
                    gcm.distribution_change,
                    self.model,
                    data_old,
                    data_new,
                    self.target,
                    num_samples=500,
                    # difference_estimation_func=lambda x1, x2: np.mean(x2) - np.mean(x1),
                ),
                num_bootstrap_resamples=5,
            )
        else:
            node_contributions = gcm.distribution_change(self.model, self.data, data_new, self.target, num_samples=500)
        node_contributions = {k: float(v) for k, v in node_contributions.items()}
        self._plot(
            basename="distribution_change_attribution",
            data=node_contributions,
            uncertainties=confidence_intervals,
            ylabel="Distribution change attribution score",
            title=f"Distribution change attribution plot for {self.data_name}",
        )
        most_impactful_node = self._get_most_impactful_node(node_contributions)
        interpretation = f"""Distribution change likelihood scores : {node_contributions}
        The node {most_impactful_node} has the highest likelihood of causing the distribution change seen in your given data.
        A negative value indicates that a node contributes to a decrease and a positive value to an increase of the mean."""
        return node_contributions, interpretation

    def get_feature_relevance(self) -> Tuple[Dict, np.ndarray, str]:
        """
        Question : How relevant is a feature for my target ?
        Examples :
        >>> parent_relevance, noise_relevance, interpretation = causal_analysis.get_feature_relevance()
        """
        logger.info(f"Calculating feature relevance for {self.target}")
        parent_relevance, noise_relevance = gcm.parent_relevance(self.model, target_node=self.target)
        most_impactful_node = self._get_most_impactful_node(parent_relevance)
        interpretation = f"""Feature relevance scores : {parent_relevance} ; Noise relevance score : {noise_relevance}.
        The relation {most_impactful_node} has the highest relevance to the target {self.target} (highest contribution to the variance of {self.target})."""
        return parent_relevance, noise_relevance, interpretation

    def _load_model_from_file(self) -> gcm.InvertibleStructuralCausalModel:
        import pickle
        from importlib.resources import open_binary

        model_filepath = f"{self.data_name}_model.pkl"
        logger.info(f"Loading model from {model_filepath}")
        with open_binary(data, model_filepath) as f:
            return pickle.load(f)

    def _plot(self, basename: str, data: Dict, ylabel: str, title: str, in_memory: bool = True, uncertainties: Optional[Dict] = None):
        from dowhy.utils import bar_plot

        image_basename = f"{basename}_{self.data_name}"
        bar_plot(data, uncertainties=uncertainties, ylabel=ylabel, display_plot=False)
        visualise_graph(image_basename, title, in_memory=in_memory)
