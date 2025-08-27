import os
from importlib import resources
from dotenv import load_dotenv
from typing import Literal, List

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
DATA_NAMES = Literal["supply_chain_medical", "supply_chain_logistics"]
