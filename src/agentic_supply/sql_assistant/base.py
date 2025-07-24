"""

Examples :
    python -m agentic_supply.sql_assistant.base --train
    python -m agentic_supply.sql_assistant.base --ask --question "What are the unique countries ?"
    python -m agentic_supply.sql_assistant.base --chat

References :
    https://github.com/vanna-ai/notebooks/blob/main/sqlite-openai-azure-vannadb.ipynb
    https://github.com/vanna-ai/vanna/blob/main/src/vanna/faiss/faiss.py
    https://github.com/vanna-ai/vanna/issues/763 issues for plotly solved by downgrading to 5.22.0

"""

import argparse
from vanna.openai import OpenAI_Chat
from openai import AzureOpenAI
from vanna.faiss import FAISS
from vanna.flask import VannaFlaskApp
# from vanna.vannadb import VannaDB_VectorStore # cannot use as Cognizant blocks the access to their domain
# from vanna.chromadb import ChromaDB_VectorStore # cannot install chromadb as it requires installing c++ build tools

from agentic_supply.utilities.config import AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, VANNA_API_KEY


class MyVanna(FAISS, OpenAI_Chat):
    def __init__(self, config: dict = {"path": "./data", "model": "gpt-4o"}):
        client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT)
        FAISS.__init__(self, config=config)
        # VannaDB_VectorStore.__init__(self, vanna_model="agentic-supply", vanna_api_key=VANNA_API_KEY, config=config)
        OpenAI_Chat.__init__(self, client=client, config=config)

    def default_train(self):
        df_ddl = self.run_sql("SELECT type, sql FROM sqlite_master WHERE sql is not null")
        print(df_ddl["sql"].to_list())
        for ddl in df_ddl["sql"].to_list():
            self.train(ddl=ddl)
        return self


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, required=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--ask", type=bool, required=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--chat", type=bool, required=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--question", type=str, required=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    vn = MyVanna()
    vn.connect_to_sqlite("data/shipment.db")
    if args.train:
        vn.default_train()
    if args.ask:
        vn.ask(question=args.question)
    if args.chat:
        app = VannaFlaskApp(vn)
        app.run()


if __name__ == "__main__":
    main()
