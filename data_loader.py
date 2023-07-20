import sqlite3
import pandas as pd

def data_loader():
    con = sqlite3.connect("./src/data/failure.db")
    data = pd.read_sql_query("SELECT * from failure", con)
    return data