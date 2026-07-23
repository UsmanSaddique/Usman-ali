import urllib.request, json, time
API="http://127.0.0.1:8000"
for i in range(40):
    try:
        with urllib.request.urlopen(API+"/api/system/preflight",timeout=10) as r:
            d=json.loads(r.read())
        print(f"UP after ~{i*3}s, preflight ok={d['ok']}", flush=True)
        for c in d['checks']:
            if not c['ok']:
                print('  FAIL', c['name'], c.get('detail',''), flush=True)
        break
    except Exception as e:
        print(f"waiting... ({type(e).__name__})", flush=True)
        time.sleep(3)
