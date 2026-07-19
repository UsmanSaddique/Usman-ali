import sqlite3
conn = sqlite3.connect('ai_director.db')
c = conn.cursor()

c.execute("SELECT id, name FROM channels WHERE name LIKE '%Ai war%'")
channels = c.fetchall()
if not channels:
    print("No channel found")
    exit()

for channel in channels:
    print(f"Channel {channel[1]} (id={channel[0]})")
    c.execute("SELECT id, status, error_log, created_at FROM projects WHERE channel_id = ? ORDER BY created_at DESC", (channel[0],))
    projects = c.fetchall()
    for p in projects[:3]:
        print(f"  Project {p[0]} (status={p[1]}) [created={p[3]}] Error: {p[2]}")
        c.execute("SELECT id, scene_number, status, scene_type FROM scenes WHERE project_id = ? ORDER BY scene_number ASC", (p[0],))
        scenes = c.fetchall()
        for s in scenes:
            print(f"    Scene {s[1]} (id={s[0]}, type={s[3]}): Status: {s[2]}")
