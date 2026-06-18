import sqlite3
conn = sqlite3.connect('ai_director.db')
c = conn.cursor()
c.execute("SELECT id, status, video_path FROM scenes WHERE project_id='6801ac66-4d60-4c14-8d46-0fb37bf3e278'")
for row in c.fetchall():
    print(row)
