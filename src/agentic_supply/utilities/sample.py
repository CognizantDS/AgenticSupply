import sqlite3
from vanna.remote import VannaDefault
from agentic_supply.utilities.config import VANNA_API_KEY

# vn = VannaDefault(model="chinook", api_key=VANNA_API_KEY)
# vn.connect_to_sqlite("https://vanna.ai/Chinook.sqlite")
# vn.ask("What are the top 10 albums by sales ?")


def build_db_():
    con = sqlite3.connect("data/tutorial.db")
    cur = con.cursor()
    cur.execute("CREATE TABLE movie(title, year, score)")
    res = cur.execute("SELECT name FROM sqlite_master")
    print(res.fetchone())

    cur.execute("""
        INSERT INTO movie VALUES
            ('Monty Python and the Holy Grail', 1975, 8.2),
            ('And Now for Something Completely Different', 1971, 7.5)
    """)
    con.commit()

    res = cur.execute("SELECT score FROM movie")
    print(res.fetchall())


vn = VannaDefault(model="agentic-supply", api_key=VANNA_API_KEY)
vn.connect_to_sqlite("data/shipment.db")
print(vn.get_models())
vn.ask("What are the top 10 albums by sales ?")
