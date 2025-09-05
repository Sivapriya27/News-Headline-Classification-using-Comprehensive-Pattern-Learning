from pathlib import Path
import json
from datetime import datetime

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def timestamp():
    return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
