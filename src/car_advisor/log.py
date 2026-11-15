import json
import sys
import time


def log(event, **data):
    line = json.dumps({"t": round(time.time(), 2), "event": event, **data}, ensure_ascii=False)
    print(line, file=sys.stderr)
