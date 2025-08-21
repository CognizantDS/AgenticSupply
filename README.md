# AgenticSupply

This repository is a python package that provides agentic AI networks for causal analysis using the Neuro-SAN and DoWhy frameworks.  

## Setting the Python venv

Clone the repository and execute the following commands to create and activate a venv, and to install the package and its dependencies (see [pyproject.toml](pyproject.toml)) :
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

NB : to install the dev dependencies as well for running tests, run ``pip install -e '.[dev]'``
NB : the ``neuro-san`` library is installed directly from git to get the latest developments (the latest pypi version as of 21st August 2025, 0.5.55, does not include fixes for Windows for resource tracking and agent thinking logs); this requires enabling the SSL certificates via ``git config --global http.sslBackend schannel``.  

## Setting the environment file
Add a .env file in this directory, with the following content :
```
AZURE_OPENAI_ENDPOINT=""
OPENAI_API_VERSION="2025-01-01-preview"
OPENAI_API_KEY=""
```

## Setting up Neuro-SAN-studio
Clone ``https://github.com/cognizant-ai-lab/neuro-san-studio`` as a separate directory.
In the ``neuro-san-studio`` directory, create and activate the venv, and install this package (-e for editable option) and the additional requirements :
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ../AgenticSupply
pip install -r requirements.txt
```

Add a .env file in the ``neuro-san-studio`` directory, with the following content :
```
AZURE_OPENAI_ENDPOINT=""
OPENAI_API_VERSION="2025-01-01-preview"
OPENAI_API_KEY=""
AGENT_TOOL_PATH="agentic_supply.supply_chain_explorer" # "../AgenticSupply/src/agentic_supply/supply_chain_explorer"
AGENT_MANIFEST_FILE="../AgenticSupply/src/agentic_supply/supply_chain_explorer/manifest.hocon"
```

## Running the Neuro-SAN-studio
Within a terminal, cd to the ``neuro-san-studio`` directory, activate its venv and run the Flask app :
```
.\.venv\Scripts\activate
python -m run
```
This will run ``neuro-san-studio`` using a venv in which the latest version of ``neuro-san`` and ``agentic_supply`` (this package) are both installed.  
It will use AGENT_TOOL_PATH to find the module corresponding to the CodedTool, and will be able to execute it using the venv.  
AGENT_MANIFEST_FILE ensures we load only the network(s) of this package for faster UI loading.  


## Info

Vanna.ai SQL Agent
 
Autogen Studio RCA Agent
 
Neuro AI - Agent Recommendation System
 
Supply Chain Opt
 