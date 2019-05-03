import json
import pickle
import random
from nltk.translate.bleu_score import sentence_bleu

from filepaths import *
from inference import get_caption

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
        rand_cap = random.choice(caps)
        img_caps.append((id, file_name, rand_cap))

    return img_caps

if __name__ == "__main__":
    index = generate_index()

    for (id, file_name, cap) in index:
        path = os.path.join(VAL_IMG_DIR, file_name)
        pred_cap = get_caption(path)
        print("Real cap = {}".format(cap))
        print("Pred cap = {}".format(pred_cap))

