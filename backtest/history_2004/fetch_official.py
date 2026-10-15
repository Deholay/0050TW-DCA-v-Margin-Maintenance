"""Public TWSE history retrieval. Two workers, paced; stop on any access block."""
import json,gzip,datetime as dt,time,threading,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'raw';RAW.mkdir(exist_ok=True)
STOP=threading.Event();LOCK=threading.Lock();COUNTS={'days':0,'trading':0,'requests':0,'cached':0}

def get(kind,day):
    for path in [RAW/f'{kind}-{day}.json.gz',RAW/f'{kind}-{day}.json',ROOT.parent/'raw/twse'/f'{kind}-{day}.json.gz']:
        if path.exists():
            data=json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes())
            with LOCK:COUNTS['cached']+=1
            return data
    if STOP.is_set():raise RuntimeError('stopped')
    endpoint,query=('MI_MARGN','selectType=ALL') if kind=='margin' else ('MI_INDEX','type=ALLBUT0999')
    url=f'https://www.twse.com.tw/exchangeReport/{endpoint}?response=json&date={day}&{query}'
    out=subprocess.run(['curl','--fail','-L','--max-time','30','-sS',url],capture_output=True)
    with LOCK:COUNTS['requests']+=1
    time.sleep(.5)
    if out.returncode:
        STOP.set();raise RuntimeError(f'{kind} {day}: request failed {out.stderr.decode()[:160]}')
    try:data=json.loads(out.stdout)
    except ValueError:
        (ROOT/f'blocked-{kind}-{day}.txt').write_bytes(out.stdout)
        STOP.set();raise RuntimeError(f'{kind} {day}: non-JSON response; stopped, no bypass/retry')
    if data.get('stat')=='OK' and data.get('date')!=day:
        STOP.set();raise RuntimeError(f'{kind} {day}: date mismatch')
    if data.get('stat')!='OK' and not any(x in str(data.get('stat')) for x in ['沒有符合','查無','無資料']):
        STOP.set();raise RuntimeError(f'{kind} {day}: unexpected status {data.get("stat")}')
    (RAW/f'{kind}-{day}.json.gz').write_bytes(gzip.compress(json.dumps(data,ensure_ascii=False).encode()))
    return data

def work(day):
    if STOP.is_set():return
    m=get('margin',day)
    if m.get('stat')=='OK':
        p=get('prices',day)
        if p.get('stat')!='OK':
            STOP.set();raise RuntimeError(f'missing price on margin day {day}')
        with LOCK:COUNTS['trading']+=1
    with LOCK:
        COUNTS['days']+=1
        if COUNTS['days']%50==0:print('PROGRESS',day,COUNTS,flush=True)

if __name__=='__main__':
    start=dt.date(2004,2,11);end=dt.date(2010,12,31)
    days=[(start+dt.timedelta(days=i)).strftime('%Y%m%d') for i in range((end-start).days+1) if (start+dt.timedelta(days=i)).weekday()!=6]
    print('START',len(days),'calendar dates excluding Sundays; Saturdays included',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(work,d) for d in days]
        for f in as_completed(futures):
            try:f.result()
            except Exception as e:
                print('STOP',str(e),flush=True);STOP.set()
                for pending in futures:pending.cancel()
                break
    print('END','BLOCKED' if STOP.is_set() else 'COMPLETE',COUNTS,flush=True)
