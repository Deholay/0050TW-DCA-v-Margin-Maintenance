"""One public TWSE request per invocation; at least 5 seconds between attempts."""
import datetime as dt,json,time,fcntl
from pathlib import Path
import fetch_official as fetch
R=Path(__file__).resolve().parent
STATE=R/'paced_state.json'

def exists(kind,day):
    return any(p.exists() for p in [R/'raw'/f'{kind}-{day}.json.gz',R/'raw'/f'{kind}-{day}.json',R.parent/'raw/twse'/f'{kind}-{day}.json.gz'])

def main():
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get('status') in ['blocked','complete']:
        print(json.dumps(state));return
    if time.time()-state.get('last_attempt_epoch',0)<5:
        print('WAIT: five-second request spacing');return
    start=dt.date(2004,2,11);end=dt.date(2010,12,31)
    target=None
    for i in range((end-start).days+1):
        date=start+dt.timedelta(days=i)
        if date.weekday()==6:continue
        day=date.strftime('%Y%m%d')
        if not exists('margin',day):target=('margin',day);break
        m=fetch.get('margin',day)
        if m.get('stat')=='OK' and not exists('prices',day):target=('prices',day);break
    if target is None:
        state['status']='complete';STATE.write_text(json.dumps(state,indent=2));print('COMPLETE');return
    state.update(status='running',last_attempt_epoch=time.time(),kind=target[0],date=target[1])
    STATE.write_text(json.dumps(state,indent=2))
    try:
        data=fetch.get(*target)
        state.update(status='ready',source_status=data.get('stat'),successful_requests=state.get('successful_requests',0)+1)
    except Exception as exc:state.update(status='blocked',error=str(exc))
    STATE.write_text(json.dumps(state,ensure_ascii=False,indent=2))
    print(json.dumps(state,ensure_ascii=False))

if __name__=='__main__':
    with (R/'paced_fetch.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('WAIT: another request is running')
        else:
            main()
