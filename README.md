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
Comment out the reference to ``neuro-san`` in the requirements.txt : this is to ensure that we install it from git as a dependency of ``agentic_supply`` :
```
# Neuro SAN
# neuro-san==0.5.55 # install via AgenticSupply
``` 

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
AGENT_TOOL_PATH="agentic_supply" # "../AgenticSupply/src/agentic_supply/supply_chain_explorer"
AGENT_MANIFEST_FILE="../AgenticSupply/src/agentic_supply/agentic_logistics/manifest.hocon"
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


## Script

1. Inventory and manufacturing (for a product) :
Can you do an inventory check for the product PURAC_FCC across all our sites please ?

Yes, please proceed to the manufacturing order scheduling.

It is to be manufactured at the Rayong Site, for 20 units, to Germany and by 20th of September 2025.

(Please directly proceed and schedule the order.)



2. Logistics - Plan a shipment route
I need to plan a shipment for a manufacturing order, please help me to plan the logistics.

From the manufacturing site "Rayong Site" to the customer facility "Henkel Facility", and for manufacturing order id "afcb40466e744b139c532b2df6186aaf". 

(Yes I confirm that there are no specific preferences, please proceed.)
OR
(Yes I confirm, please proceed to shipment route planning.)

(The Henkel Facility is in Germany. Please proceed directly to route generation.)

Great ! Let's use "Option 3: Balanced Scenario".



3. Logistics - Place a shipment
I need to place a shipment with the following details :
manufacturing order id is "afcb40466e744b139c532b2df6186aaf" ;
land route ids are 1, 2 ; ocean route ids are 1

Yes, I confirm.


4. Disaster recovery
Are there any issues impacting shipment delivery ?

Yes please, propose rerouting options for shipment id 43d5c725b1b144908e10573a6634c543. Please use the data of the shipment to directly propose new routes.

Please place the shipment order for Route ID: 2

Yes I confirm.