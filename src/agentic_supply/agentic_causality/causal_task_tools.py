from typing import Any, Dict
from neuro_san.interfaces.coded_tool import CodedTool
import os
import pandas as pd

from agentic_supply.causality_assistant.causal_analysis import CausalAnalysis
from agentic_supply.utilities.config import DATA_NAMES, CAUSAL_INFLUENCE_TYPES, ROOT_CAUSE_TYPES, WHAT_IF_QUESTION_TYPES
from agentic_supply.data_assistant.data_downloading import select_target_path
from agentic_supply.utilities.log_utils import set_logging, get_logger


set_logging()
logger = get_logger(__name__)


class CausalInfluenceQuantificator(CodedTool):
    """
    CodedTool implementation of a calculator for the math_guy test.

    Upon activation by the agent hierarchy, a CodedTool will have its
    invoke() call called by the system.

    Implementations are expected to clean up after themselves.
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """
        Called when the coded tool is invoked asynchronously by the agent hierarchy.
        Strongly consider overriding this method instead of the "easier" synchronous
        version above when the possibility of making any kind of call that could block
        (like sleep() or a socket read/write out to a web service) is within the
        scope of your CodedTool.

        :param args: An argument dictionary whose keys are the parameters
                to the coded tool and whose values are the values passed for them
                by the calling agent.  This dictionary is to be treated as read-only.
        :param sly_data: A dictionary whose keys are defined by the agent hierarchy,
                but whose values are meant to be kept out of the chat stream.

                This dictionary is largely to be treated as read-only.
                It is possible to add key/value pairs to this dict that do not
                yet exist as a bulletin board, as long as the responsibility
                for which coded_tool publishes new entries is well understood
                by the agent chain implementation and the coded_tool implementation
                adding the data is not invoke()-ed more than once.
        :return: A return value that goes into the chat stream.
        """

        data_name: DATA_NAMES = sly_data["data_name"]
        logger.info(f"data_name from sly_data : {data_name}")
        causal_influence_type: CAUSAL_INFLUENCE_TYPES = args.get("causal_influence_type")
        logger.info(f"causal_influence_type from args : {causal_influence_type}")

        causal_analysis = CausalAnalysis.from_file(data_name)
        if causal_influence_type == "arrow":
            result = causal_analysis.get_arrow_strength()
        elif causal_influence_type == "intrinsic":
            result = causal_analysis.get_intrinsic_causal_influence()
        else:
            raise ValueError("invalid causal_influence_type")

        return f"The causal influence was correctly quantified, see the node-wise result :\n{result}"


class RootCauseAnalyser(CodedTool):
    """
    CodedTool implementation of a calculator for the math_guy test.

    Upon activation by the agent hierarchy, a CodedTool will have its
    invoke() call called by the system.

    Implementations are expected to clean up after themselves.
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """
        Called when the coded tool is invoked asynchronously by the agent hierarchy.
        Strongly consider overriding this method instead of the "easier" synchronous
        version above when the possibility of making any kind of call that could block
        (like sleep() or a socket read/write out to a web service) is within the
        scope of your CodedTool.

        :param args: An argument dictionary whose keys are the parameters
                to the coded tool and whose values are the values passed for them
                by the calling agent.  This dictionary is to be treated as read-only.
        :param sly_data: A dictionary whose keys are defined by the agent hierarchy,
                but whose values are meant to be kept out of the chat stream.

                This dictionary is largely to be treated as read-only.
                It is possible to add key/value pairs to this dict that do not
                yet exist as a bulletin board, as long as the responsibility
                for which coded_tool publishes new entries is well understood
                by the agent chain implementation and the coded_tool implementation
                adding the data is not invoke()-ed more than once.
        :return: A return value that goes into the chat stream.
        """

        data_name: DATA_NAMES = sly_data["data_name"]
        logger.info(f"data_name from sly_data : {data_name}")
        root_cause_type: ROOT_CAUSE_TYPES = args.get("root_cause_type")
        logger.info(f"root_cause_type from args : {root_cause_type}")

        causal_analysis = CausalAnalysis.from_file(data_name)

        if root_cause_type != "feature_relevance":
            data_path = select_target_path("openname")
            logger.info(f"data_path selected : {data_path}")
            data_new = pd.read_csv(data_path)

        if root_cause_type == "anomaly_attributon":
            node_contributions, interpretation = causal_analysis.get_anomaly_attribution(anomalous_data=data_new)
        elif root_cause_type == "distribution_attribution":
            node_contributions, interpretation = causal_analysis.get_distribution_change_attribution(data_new=data_new)
        elif root_cause_type == "feature_relevance":
            node_contributions, noise_relevance, interpretation = causal_analysis.get_feature_relevance()
        else:
            raise ValueError("invalid root_cause_type")

        return (
            f"The root cause was correctly analysed, see the node-wise result :\n{node_contributions}\nInterpretation :\n{interpretation}"
        )


class WhatIfAnswerer(CodedTool):
    """
    CodedTool implementation of a calculator for the math_guy test.

    Upon activation by the agent hierarchy, a CodedTool will have its
    invoke() call called by the system.

    Implementations are expected to clean up after themselves.
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """
        Called when the coded tool is invoked asynchronously by the agent hierarchy.
        Strongly consider overriding this method instead of the "easier" synchronous
        version above when the possibility of making any kind of call that could block
        (like sleep() or a socket read/write out to a web service) is within the
        scope of your CodedTool.

        :param args: An argument dictionary whose keys are the parameters
                to the coded tool and whose values are the values passed for them
                by the calling agent.  This dictionary is to be treated as read-only.
        :param sly_data: A dictionary whose keys are defined by the agent hierarchy,
                but whose values are meant to be kept out of the chat stream.

                This dictionary is largely to be treated as read-only.
                It is possible to add key/value pairs to this dict that do not
                yet exist as a bulletin board, as long as the responsibility
                for which coded_tool publishes new entries is well understood
                by the agent chain implementation and the coded_tool implementation
                adding the data is not invoke()-ed more than once.
        :return: A return value that goes into the chat stream.
        """

        data_name: DATA_NAMES = sly_data["data_name"]
        logger.info(f"data_name from sly_data : {data_name}")
        what_if_question_type: WHAT_IF_QUESTION_TYPES = args.get("what_if_question_type")
        logger.info(f"what_if_question_type from args : {what_if_question_type}")
        node: str = args.get("node")
        logger.info(f"node from args : {node}")

        causal_analysis = CausalAnalysis.from_file(data_name)
        if what_if_question_type == "intervention":
            df = causal_analysis.generate_interventional_samples(node=node)
        elif what_if_question_type == "counterfactual":
            df = causal_analysis.generate_counterfactual_samples(node=node)
        else:
            raise ValueError("invalid what_if_question_type")

        return f"The what-if question was correctly answered, please see the answer in the form of a generated dataframe below :\n{df}"
