import sqlite3, json, time, urllib.request, glob
PID='58ed10a3-c73c-4e91-9c8f-730b711c329f'
TERM={'RENDERED','COMPLETE','COMPLETED','FAILED','UPLOADED','DONE'}
js=f'projects/{PID}/run_state.json'
API="http://127.0.0.1:8000"
last=''
def alive():
    try:
        urllib.request.urlopen(API+"/api/system/gpu-status",timeout=6); return True
    except: return False
while True:
    srv=alive()
    try:
        con=sqlite3.connect('ai_director.db')
        st=con.execute('select status from projects where id=?',(PID,)).fetchone()[0]
        con.close()
    except Exception: st='?'
    phase=msg='';pct=None
    try:
        d=json.load(open(js)); phase=d.get('phase');msg=(d.get('message') or '')[:55];pct=d.get('percent')
    except: pass
    parts=len(glob.glob(f'projects/{PID}/ltx_parts/part_*.mp4'))
    line=f"{time.strftime('%H:%M:%S')} srv={'UP' if srv else 'DOWN'} status={st} phase={phase} pct={pct} parts={parts}/10 :: {msg}"
    if line[9:]!=last[9:]:
        print(line, flush=True); last=line
    if not srv:
        print('!! SERVER DOWN — pipeline died (resume: parts are saved)', flush=True); break
    if str(st).upper() in TERM:
        con=sqlite3.connect('ai_director.db')
        op=con.execute('select output_path,error_log from projects where id=?',(PID,)).fetchone()
        print('TERMINAL:',st,'output=',op[0],'err=',op[1], flush=True); con.close()
        break
    time.sleep(45)
