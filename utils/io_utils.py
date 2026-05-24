import json
import numpy as np

def save_json(data, path):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(path):

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_numpy(array, path):

    np.save(path, array)

def load_numpy(path):
    
    return np.load(path)