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
AGENT_TOOL_PATH="agentic_supply" 
AGENT_MANIFEST_FILE="../AgenticSupply/src/agentic_supply/manifest.hocon"
```

To get and save orders and shipments, we need to add two more files within the ``neuro-san-studio/logs`` directory :
- ``order_db.json`` :
```
{
    "orders": []
}
```
- ``shipment_db.json`` :
```
{
    "shipments": []
}
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
Any orders and shipments generated will be stored in ``neuro-san-studio/logs/order_db.json`` and ``neuro-san-studio/logs/shipment_db.json`` respectively.  

## Script
### Agentic Causality
#### 0. Choose data name
Hi
-> Prompts to select a dataset to use for the duration of the chat

example_data
-> Stores this as a string in sly_data, and prompts to choose a task

#### 1. Causal Graph
visualise the causal graph
-> Opens the plot in a new webbrowser tab

refute the causal graph
-> Opens the plot in a new webbrowser tab, returns text interpretation (LONG !)

#### 2. Causal Model
evaluate the graphical causal model (LONG !)  <PB: does not return interpretation to the chat>
-> Returns text interpretation

generate data from the graphical causal model (for X samples)  
-> Uses OS to save and open the dataframe

#### 3. Causal Tasks
##### Quantifying causal influence
quantify causal influence via arrow strength
-> Returns text interpretation

quantify causal influence via the intrinsic mode
-> Opens the plot in a new webbrowser tab, returns text interpretation

##### Performing root cause analysis
perform root cause analysis via the anomaly attribution mode   <PB: interpretation is not formatted, only sometimes (prompt engineering ?)>
-> Uses OS to prompt the user to select the anomalous dataset csv file, opens the plot in a new webbrowser tab, returns text interpretation

perform root cause analysis via the distribution change attribution mode (LONG !)
-> Uses OS to prompt the user to select the novel dataset csv file, opens the plot in a new webbrowser tab, returns text interpretation

perform root cause analysis via the feature importance mode
-> Returns text interpretation

##### Answering causal what-if questions
answer causal what-if questions via the intervention mode, with intervention x * 100 on node X
-> Uses OS to save and open the dataframe

answer causal what-if questions via the counterfactual mode, with intervention 0 on node Y 
-> Uses OS to prompt the user to select the dataset from which to generate counterfactuals, Uses OS to save and open the dataframe

### 4. Use cases
#### Finding Root Causes of Changes in a Supply Chain
Reproduce [reference notebook](https://www.pywhy.org/dowhy/v0.13/example_notebooks/gcm_supply_chain_dist_change.html) via the chat.  

Hi
-> Prompts to select a dataset to use for the duration of the chat

supply_chain_logistics
-> Stores this as a string in sly_data, and prompts to choose a task

visualise the causal graph
-> Opens the plot in a new webbrowser tab

perform root cause analysis via the distribution change attribution mode (hardcoded to use week 1 and week 2 as comparison datasets)
-> Uses OS to prompt the user to select the novel dataset csv file, opens the plot in a new webbrowser tab, returns text interpretation

### Agentic Logistics
#### 1. Inventory and manufacturing (for a product) :
Can you do an inventory check for the product PURAC_FCC across all our sites please ?  
-> Returns data and prompts the user to continue ordering for Rayong Site  
  
Yes, please proceed to the manufacturing order scheduling.  
-> Prompts the user for the additional required details of quantity, destination and date.  

It is to be manufactured at the Rayong Site, for 20 units, to Germany and by 20th of September 2025.  
-> Either proceeds directly or asks confirmation.  
  
(Please directly proceed and schedule the order.)  
-> Returns order details  

#### 2. Logistics - Plan a shipment route
I need to plan a shipment for a manufacturing order, please help me to plan the logistics by generating door-to-door routing options.  
-> Prompts the user for the additional required details of manufacturing site and customer facility.  
  
From the manufacturing site "Rayong Site" to the customer facility "Henkel Facility".
-> Either proceeds directly or asks confirmation   

(Yes I confirm, please proceed to shipment route planning.)  
-> Returns 3 options and prompts the user to choose one.  

#### 3. Logistics - Place a shipment
I need to place a shipment for a manufacturing order. What information do you need ?
-> Prompts the user for additional required details of manufacturing order id, land route ids and ocean route ids.

The manufacturing order id is "afcb40466e744b139c532b2df6186aaf" ; land route ids are 2, 12 ; ocean route ids are 1  
-> Either proceeds directly or asks confirmation 
NB : Choose an order id existing in order_db.json (or the one you generated during the chat)   
  
(Yes, I confirm. Please proceed to the shipment placement directly.) 
-> Returns shipment details  

#### 4. Disaster recovery
Are there any issues impacting shipment delivery ?  
-> Returns an issue for shipments going through Singapore. Prompts the user to reroute the shipment.  
  
Yes please, propose rerouting options for shipment id 441cbc3bc93243c9a4068377b984a279. With origin, the manufacturing site "Rayong Site", and destination, the customer facility "Henkel Facility". 
-> Returns 3 options and prompts the user to choose one.  
NB : Choose a shipment id existing in shipment_db.json (or the one you generated during the chat) 

(Yes please, directly return to me the alternative routes.)
