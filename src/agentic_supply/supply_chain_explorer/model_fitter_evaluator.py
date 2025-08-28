from typing import Any, Dict
from neuro_san.interfaces.coded_tool import CodedTool
import webbrowser
import os
import pickle

from agentic_supply.causality_assistant.causal_model_fitting_evaluation import fit_eval_causal_model
from agentic_supply.utilities.config import DATA_NAMES


class ModelFitterEvaluator(CodedTool):
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

        data_name: DATA_NAMES = args.get("data_name")
        causal_graph_path = sly_data.get("causal_graph_path")

        with open(causal_graph_path, "rb") as file:
            causal_graph = pickle.load(file)

        causal_model, evaluation, image_filepath_html = fit_eval_causal_model(data_name, causal_graph)
        sly_data["causal_model"] = causal_model
        sly_data["evaluation"] = evaluation

        webbrowser.open_new_tab(f"file://{os.path.abspath(image_filepath_html)}")

        return "See the causal model evaluation graph in the newly opened tab"
