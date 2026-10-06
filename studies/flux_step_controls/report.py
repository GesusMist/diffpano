from .common import *

def main():
    result=[]
    for r in rows():
        status=ROOT/'status'/(str(r['index'])+'.json')
        s='complete' if complete(r) else read(status)['state'] if status.exists() else 'missing_or_pending'
        result.append(dict(row=r,state=s))
    atomic(ROOT/'completion.json',dict(time=now(),rows=result,
        complete=sum(r['state']=='complete' for r in result),expected=42))
    emit('CONTROL_COMPLETION',rows=result)
if __name__=='__main__':main()
