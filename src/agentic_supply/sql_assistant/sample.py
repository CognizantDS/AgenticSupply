from vanna.remote import VannaDefault
from agentic_supply.utilities.config import VANNA_API_KEY

# vn = VannaDefault(model="chinook", api_key=VANNA_API_KEY)
# vn.connect_to_sqlite("https://vanna.ai/Chinook.sqlite")
# vn.ask("What are the top 10 albums by sales ?")

vn = VannaDefault(model="agentic-supply", api_key=VANNA_API_KEY)
vn.connect_to_sqlite("data/shipment.db")
print(vn.get_models())
vn.ask("What are the top 10 albums by sales ?")
