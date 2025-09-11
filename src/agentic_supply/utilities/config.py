import os
from importlib import resources
from dotenv import load_dotenv
from typing import Literal, List, Dict, Tuple

load_dotenv(override=True)

# azure openai
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_VERSION_MINI = os.getenv("AZURE_OPENAI_API_VERSION_MINI")
AZURE_OPENAI_NAME_4O = os.getenv("AZURE_OPENAI_NAME_4O")
AZURE_OPENAI_NAME_4O_MINI = os.getenv("AZURE_OPENAI_NAME_4O_MINI")

# azure ai
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT")
AZURE_AI_KEY = os.getenv("AZURE_AI_KEY")
AZURE_AI_API_VERSION = os.getenv("AZURE_AI_API_VERSION")
AZURE_AI_NAME_MISTRAL = os.getenv("AZURE_AI_NAME_MISTRAL")
AZURE_AI_NAME_GROK = os.getenv("AZURE_AI_NAME_GROK")
AZURE_AI_NAME_LLAMA = os.getenv("AZURE_AI_NAME_LLAMA")
AZURE_AI_NAME_PHI = os.getenv("AZURE_AI_NAME_PHI")
AZURE_AI_NAME_DEEPSEEK = os.getenv("AZURE_AI_NAME_DEEPSEEK")

# vanna ai
VANNA_API_KEY = os.getenv("VANNA_API_KEY")

# constants
DATA_NAMES = Literal["example_data", "supply_chain_medical", "supply_chain_logistics"]
DATA_TO_FILE: Dict[DATA_NAMES, List[Tuple]] = {
    "example_data": "example_data.csv",
    "online_shop_data": "online_shop_data.csv",
    "supply_chain_medical": "SCMS_Delivery_History_Dataset.csv",
    "supply_chain_logistics": "supply_chain_week_over_week.csv",
}
DATA_TO_TARGET: Dict[DATA_NAMES, List[Tuple]] = {
    "example_data": "Z",
    "online_shop_data": "Profit",
    "supply_chain_medical": "",
    "supply_chain_logistics": "received",
}
CAUSAL_INFLUENCE_TYPES = Literal["intrinsic", "arrow"]
ROOT_CAUSE_TYPES = Literal["anomaly_attributon", "distribution_attribution", "feature_relevance"]
WHAT_IF_QUESTION_TYPES = Literal["intervention", "counterfactual"]
PRODUCT_NAMES = Literal["lactic_acid", "ascorbic_acid"]
DESTINATIONS = Literal["Germany", "Netherlands", "Belgium", "Denmark"]
ARTIFACTS_DIR = "./logs"
