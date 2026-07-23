import sqlite3, json, time, os
PID='ab07acf9-2786-4bbb-b742-0ad2892684f4'
TERM={'RENDERED','COMPLETE','COMPLETED','FAILED','UPLOADED','DONE'}
js=f'projects/{PID}/run_state.json'
last=''
while True:
    try:
        con=sqlite3.connect('ai_director.db')
        st,tot,done=con.execute('select status,total_scenes,completed_scenes from projects where id=?',(PID,)).fetchone()
        con.close()
    except Exception as e:
        st='?';tot=0;done=0
    phase=msg=''
    try:
        d=json.load(open(js)); phase=d.get('phase');msg=(d.get('message') or '')[:70];pct=d.get('percent')
    except: pct=None
    line=f"{time.strftime('%H:%M:%S')} status={st} scenes={done}/{tot} phase={phase} pct={pct} :: {msg}"
    if line[9:]!=last[9:]:
        print(line, flush=True)
        last=line
    if str(st).upper() in TERM:
        print('TERMINAL:', st, flush=True); break
    time.sleep(20)
