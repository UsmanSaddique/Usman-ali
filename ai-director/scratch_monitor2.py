import sqlite3, json, time, urllib.request
PID='cf52e566-cb9b-4676-b7cb-918f6094e05e'
TERM={'RENDERED','COMPLETE','COMPLETED','FAILED','UPLOADED','DONE'}
js=f'projects/{PID}/run_state.json'
API="http://127.0.0.1:8000"
last=''
def alive():
    try:
        urllib.request.urlopen(API+"/api/system/gpu-status",timeout=6); return True
    except: return False
while True:
    srv = alive()
    try:
        con=sqlite3.connect('ai_director.db')
        st,tot,done=con.execute('select status,total_scenes,completed_scenes from projects where id=?',(PID,)).fetchone()
        con.close()
    except Exception: st='?';tot=0;done=0
    phase=msg='';pct=None
    try:
        d=json.load(open(js)); phase=d.get('phase');msg=(d.get('message') or '')[:60];pct=d.get('percent')
    except: pass
    line=f"{time.strftime('%H:%M:%S')} srv={'UP' if srv else 'DOWN'} status={st} scenes={done}/{tot} phase={phase} pct={pct} :: {msg}"
    if line[9:]!=last[9:]:
        print(line, flush=True); last=line
    if not srv:
        print('!! SERVER DOWN — pipeline died again', flush=True); break
    if str(st).upper() in TERM:
        con=sqlite3.connect('ai_director.db')
        op=con.execute('select output_path,error_log from projects where id=?',(PID,)).fetchone()
        print('TERMINAL:',st,'output=',op[0],'err=',op[1], flush=True); con.close()
        break
    time.sleep(30)
