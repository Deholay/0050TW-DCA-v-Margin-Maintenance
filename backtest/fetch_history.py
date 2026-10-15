import subprocess
import concurrent.futures as cf, datetime as dt, gzip, json, pathlib, time, urllib.request
ROOT=pathlib.Path(__file__).resolve().parent
CACHE=ROOT/'raw'/'twse';CACHE.mkdir(exist_ok=True)
y=json.loads((ROOT/'raw'/'0050-yahoo2.json').read_text())['chart']['result'][0]
dates=sorted({dt.datetime.fromtimestamp(t,dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d') for t in y['timestamp']})
dates=[d for d in dates if '20100101'<=d<='20261014']

def fetch(day):
    out=[]
    for kind,endpoint,query in [('margin','MI_MARGN','selectType=ALL'),('prices','MI_INDEX','type=ALLBUT0999')]:
        path=CACHE/f'{kind}-{day}.json.gz'
        if path.exists():continue
        route='marginTrading' if kind=='margin' else 'afterTrading'
        url=f'https://www.twse.com.tw/rwd/zh/{route}/{endpoint}?response=json&date={day}&{query}'
        for attempt in range(2):
            try:
                req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0','Accept-Encoding':'identity'})
                raw=subprocess.check_output(['curl','-f','-L','--max-time','20','-sS',url],timeout=25)
                data=json.loads(raw)
                if data.get('stat')=='OK' and data.get('date')!=day:raise ValueError('date mismatch')
                with gzip.open(path,'wb') as f:f.write(raw)
                break
            except Exception as e:
                if attempt==1:out.append(f'{kind}:{e}')
                else:time.sleep(2**attempt)
        time.sleep(.12)
    return day,out
start=time.monotonic()
print('START',len(dates),'days',dates[0],dates[-1],flush=True)
with cf.ThreadPoolExecutor(max_workers=4) as pool:
    for i,(day,errors) in enumerate(pool.map(fetch,dates),1):
        if errors:print('ERROR',day,errors,flush=True)
        if i%100==0:print('PROGRESS',i,'/',len(dates),'date',day,'elapsed',round(time.monotonic()-start),flush=True)
print('DONE',round(time.monotonic()-start),flush=True)
