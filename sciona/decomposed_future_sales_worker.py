"""Run one numerical training stage in a fresh process using finite JSON pipes."""
import json
import sys
from sciona.decomposed_future_sales_transport import decode_panel


def run(request):
    if not isinstance(request,dict) or set(request)!={'operation','panel','config','categorical_columns','lower','upper','selection'}:
        raise ValueError('Exact worker request required')
    from sciona.decomposed_future_sales_training import search,refit
    panel=decode_panel(request['panel'])
    args=(panel,request['config'],request['categorical_columns'],request['lower'],request['upper'])
    if request['operation']=='search' and request['selection'] is None:return search(*args)
    if request['operation']=='refit':return refit(*args,request['selection'])
    raise ValueError('Unsupported worker operation')


if __name__=='__main__':
    try:
        result=run(json.load(sys.stdin))
        sys.stdout.write(json.dumps(result,allow_nan=False))
    except Exception:
        sys.stderr.write('Isolated training failed\n')
        raise SystemExit(1) from None
