"""

References :
    https://github.com/vanna-ai/notebooks/blob/main/sqlite-openai-azure-vannadb.ipynb
    https://github.com/vanna-ai/vanna/blob/main/src/vanna/faiss/faiss.py

"""

from vanna.openai import OpenAI_Chat
from openai import AzureOpenAI
from vanna.faiss import FAISS
# from vanna.vannadb import VannaDB_VectorStore # cannot use as Cognizant blocks the access to their domain
# from vanna.chromadb import ChromaDB_VectorStore # cannot install chromadb as it requires installing c++ build tools

from agentic_supply.utilities.config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, VANNA_API_KEY


class MyVanna(FAISS, OpenAI_Chat):
    def __init__(self, config: dict = {"path": "./data", "model": "gpt-4o"}):
        client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT)
        FAISS.__init__(self, config=config)
        # VannaDB_VectorStore.__init__(self, vanna_model="agentic-supply", vanna_api_key=VANNA_API_KEY, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config)


if __name__ == "__main__":
    vn = MyVanna()
    vn.connect_to_sqlite("data/shipment.db")
    df_ddl = vn.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
    # print(df_ddl["sql"].to_list())
    # for ddl in df_ddl["sql"].to_list():
    #     vn.train(ddl=ddl)
    # training_data = vn.get_training_data()
    # print(training_data)
    print(vn.ask(question="What are the unique countries ?"))
