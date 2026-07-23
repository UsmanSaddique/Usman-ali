import urllib.request, json, time, sqlite3
API="http://127.0.0.1:8000"
def get(p):
    with urllib.request.urlopen(API+p,timeout=15) as r: return json.loads(r.read())
for i in range(60):
    try:
        # engine-status or gpu-status won't tell busy; use a lightweight probe: try cancel-idempotent status
        con=sqlite3.connect('ai_director.db')
        d=con.execute("select status from projects where id=?",('ab07acf9-2786-4bbb-b742-0ad2892684f4',)).fetchone()[0]
        con.close()
    except Exception as e:
        d="?"
    # probe busy by attempting a harmless 409-guarded call is heavy; instead read run_state pid liveness
    print(f"{time.strftime('%H:%M:%S')} duck_status={d}", flush=True)
    if d not in ("SCRIPTED",):  # once it moves past or settles
        pass
    time.sleep(5)
    # try starting brush; if not busy it will start
    try:
        req=urllib.request.Request(API+"/api/projects/cf52e566-cb9b-4676-b7cb-918f6094e05e/full-auto",data=b"",method="POST",headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req,timeout=30) as r:
            body=json.loads(r.read())
        print("START RESP:",body,flush=True)
        if body.get("status")=="started":
            print("BRUSH STARTED",flush=True); break
    except urllib.error.HTTPError as e:
        msg=e.read().decode()
        if "busy" in msg.lower():
            print("still busy...",flush=True)
        else:
            print("ERR",e.code,msg,flush=True); break
