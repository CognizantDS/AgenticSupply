"""
Reference :
    https://www.kaggle.com/code/divyeshardeshana/supply-chain-shipment-price-data-analysis/input
    https://catalog.data.gov/dataset/supply-chain-shipment-pricing-data-07d29 --> original, but broken download link
"""

import sqlite3
import pandas


def build_db():
    con = sqlite3.connect("data/shipment.db")
    df = pandas.read_csv("data/SCMS_Delivery_History_Dataset.csv")
    df.to_sql(name="shipment", con=con, index=False)


def check_db():
    con = sqlite3.connect("data/shipment.db")
    cur = con.cursor()
    res = cur.execute("SELECT name FROM sqlite_master")
    res2 = cur.execute("SELECT DISTINCT Country FROM shipment")
    print(res.fetchone())
    print(res2.fetchall())


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


if __name__ == "__main__":
    # build_db()
    check_db()
