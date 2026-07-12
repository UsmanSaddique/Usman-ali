import sqlite3, json

conn = sqlite3.connect('ai_director.db')
cur = conn.cursor()
cur.execute('SELECT g.parameters, g.generation_time_sec FROM generations g WHERE g.model_used LIKE "%ltx%" AND g.generation_time_sec < 60 AND g.parameters IS NOT NULL AND g.parameters != "{}" LIMIT 5')
rows = cur.fetchall()
for r in rows:
    print(f"Time: {r[1]}, Params: {r[0]}")
