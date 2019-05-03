import json
import pickle
import random
from nltk.translate.bleu_score import sentence_bleu

from filepaths import *
from inference import get_caption
from vocab import Vocabulary

special_toks = ["<pad>", "<start>", "<end>", "<unk>"]

def compute_bleu(img, cap, n=4):
    weights = (1.0 / n,) * n
    cap_pred = get_caption(img)
    cap_pred = [c for c in cap_pred.split() if c not in special_toks]
    score = sentence_bleu([cap], cap_pred, weights=weights) 


def generate_index():
    with open(VAL_CAP_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    imgs = data["images"]
    caps = data["annotations"]

    print("Generating Index...")
    index = {}
    for c in caps:
        id = c["image_id"]
        if id not in index:
            index[id] = list()
        index[id].append(c["caption"])

    img_caps = list()

    for i in imgs: 
        id = i["id"]
        file_name = i["file_name"]
        caps = index[id]
        caps.sort(key=len)
        img_caps.append((id, file_name, caps[0]))

    return img_caps

if __name__ == "__main__":

    results_path = os.path.join(DATA_PATH, "results.pkl")
    
    if not os.path.exists(results_path):
        results = {}
        index = generate_index()
        for (id, file_name, cap) in index:
            path = os.path.join(VAL_IMG_DIR, file_name)
            pred_cap = get_caption(path)
            results[id] = (pred_cap, cap)
        
        with open(results_path, "wb") as f:
            pickle.dump(results, f)

    else:
        with open(results_path, "rb") as f:
            results = pickle.load(f)

    for (pred_cap, real_cap) in results:
        print(prd_cap)
        print(real_cap)
