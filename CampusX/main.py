import random
from fastmcp import FastMCP


#Create a FastMCP server Instance
mcp=FastMCP(name="Demo Server")

@mcp.tool
def add_numbers(a:float,b:float)->float:
    """Add two numbers together"""
    return a+b

from fastmcp import FastMCP
import os
import sqlite3


DB_PATH=os.path.join(os.path.dirname(__file__),"expenses.db")

mcp=FastMCP("ExpenseTracker")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
CREATE TABLE IF NOT EXISTS expenses(
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     date TEXT NOT NULL,
                     amount REAL NOT NULL,
                     category TEXT NOT NULL,
                     subcategory TEXT DEFAULT '',
                     note TEXT DEFAULT '')
                     """)
init_db()

@mcp.tool()
def add_expense(date,amount,category,subcatrgory="",note=""):
    with sqlite3.connect(DB_PATH) as conn:
        cursor=conn.execute(
            "INSERT INTO expenses(date,amount,category,subcategory,note) VALUES (?,?,?,?,?)",(date,amount,category,subcatrgory,note)
        )
        return {"status":"ok","id":cursor.lastrowid}
@mcp.tool
def list_expenses(start_date, end_date):
    '''List expense entries within an inclusive date range.'''
    with sqlite3.connect(DB_PATH) as c:
        cur = c.execute(
            """
            SELECT id, date, amount, category, subcategory, note
            FROM expenses
            WHERE date BETWEEN ? AND ?
            ORDER BY id ASC
            """,
            (start_date, end_date)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
 

if __name__=="__main__":
    mcp.run()

