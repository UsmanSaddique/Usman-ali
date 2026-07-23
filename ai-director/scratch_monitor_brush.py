import sqlite3, json, time
PID='cf52e566-cb9b-4676-b7cb-918f6094e05e'
TERM={'RENDERED','COMPLETE','COMPLETED','FAILED','UPLOADED','DONE'}
js=f'projects/{PID}/run_state.json'
last=''
while True:
    try:
        con=sqlite3.connect('ai_director.db')
        st,tot,done=con.execute('select status,total_scenes,completed_scenes from projects where id=?',(PID,)).fetchone()
        con.close()
    except Exception: st='?';tot=0;done=0
    phase=msg='';pct=None
    try:
        d=json.load(open(js)); phase=d.get('phase');msg=(d.get('message') or '')[:70];pct=d.get('percent')
    except: pass
    line=f"{time.strftime('%H:%M:%S')} status={st} scenes={done}/{tot} phase={phase} pct={pct} :: {msg}"
    if line[9:]!=last[9:]:
        print(line, flush=True); last=line
    if str(st).upper() in TERM:
        print('TERMINAL:', st, flush=True)
        try:
            con=sqlite3.connect('ai_director.db')
            op=con.execute('select output_path,error_log from projects where id=?',(PID,)).fetchone()
            print('output_path=',op[0]); print('error_log=',op[1]); con.close()
        except Exception as e: print('final read err',e)
        break
    time.sleep(30)
