import os
import json

def load_state(state_file: str):
    if os.path.exists(state_file):
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_state(state: dict, state_file: str):
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)