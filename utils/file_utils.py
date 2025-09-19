import os, json

def list_pdfs(pdf_folder: str):
    return {f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")}

def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_prompt(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None