import json
from filepaths import *
from inference import get_caption
from vocab import Vocabulary

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

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

    return index, img_caps

def evaluate():
    coco = COCO(VAL_CAP_FILE)
    coco_res = coco.loadRes(RESULTS_FILE)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.params["image_id"] = coco_res.getImgIds()
    coco_eval.evaluate()

if __name__ == "__main__":
    index, img_caps = generate_index()
    results_path = os.path.join(DATA_PATH, "results.json")
    
    if not os.path.exists(results_path):
        i = 0
        num_res = 4000
        results = list() 
        for (id, file_name, cap) in img_caps:
            i += 1
            path = os.path.join(VAL_IMG_DIR, file_name)
            try:
                pred_cap = get_caption(path)
            except RuntimeError:
                print("ERROR!")
                continue

            pred_cap = " ".join(pred_cap.split()[1:-2]) 
            results.append({"image_id": id, "caption": pred_cap})
            if i >= num_res: break
        
        with open(results_path, "w") as f:
            json.dump(results, f)

    evaluate()
