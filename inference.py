import json
import torch
import numpy as np
from collections import defaultdict
from evaluate import evaluate_hico_official, _load_rare_ids

def run_yolo_detector(yolo_model, image, det_score_thresh=0.3):
    """Wraps your YOLO model. Adjust only this function if its API differs."""
    raw_dets = yolo_model.predict(image)  # <-- point at your actual YOLO call
    return [d for d in raw_dets if d['score'] >= det_score_thresh]


def build_candidate_pairs(detections, hoi_to_obj, person_class_id=PERSON_CLASS_ID):
    """
    From a flat list of YOLO detections for one image, build candidate
    (human_box, object_box, candidate_hoi_ids) triples: every human paired
    with every object, restricted to the HOI classes whose object category
    (per your existing hoi_to_obj mapping) matches that object's YOLO class.
    """
    humans = [d for d in detections if d['class_id'] == person_class_id]
    objects = [d for d in detections if d['class_id'] != person_class_id]
 
    obj_class_to_hoi_ids = defaultdict(list)
    for hoi_id, obj_class_id in hoi_to_obj.items():
        obj_class_to_hoi_ids[obj_class_id].append(hoi_id)
 
    pairs = []
    for h in humans:
        for o in objects:
            candidate_hoi_ids = obj_class_to_hoi_ids.get(o['class_id'], [])
            if not candidate_hoi_ids:
                continue
            pairs.append((h['box'], o['box'], candidate_hoi_ids, h['score'] * o['score']))
    return pairs


def run_inference_yolo(model, image_items, yolo_model, device, hoi_to_obj,
                        preprocess_pair, det_score_thresh=0.3, num_hoi=600):
    """
    image_items: iterable of (image_id, PIL_or_tensor_image) covering the full test split.
    preprocess_pair: callable(image, human_box, object_box) -> (img_t, hb_t, ob_t)
    Returns a flat predictions list in the CELL A format.
    """
    model.eval()
    predictions = []
 
    with torch.no_grad():
        for image_id, image in image_items:
            detections = run_yolo_detector(yolo_model, image, det_score_thresh)
            pairs = build_candidate_pairs(detections, hoi_to_obj)
            
            if not pairs:
                continue

            # BATCH ALL PAIRS FOR MASSIVE SPEEDUP
            img_list, hb_list, ob_list = [], [], []
            for h_box, o_box, _, _ in pairs:
                img_t, hb_t, ob_t = preprocess_pair(image, h_box, o_box)
                img_list.append(img_t)
                hb_list.append(hb_t)
                ob_list.append(ob_t)
            
            # Stack into a single batch for this image
            img_batch = torch.stack(img_list).to(device)
            hb_batch  = torch.stack(hb_list).to(device)
            ob_batch  = torch.stack(ob_list).to(device)
 
            # ONE single forward pass for all candidates in the image
            out = model(img_batch, hb_batch, ob_batch, use_gt_boxes=False)
            
            # --- ABLATION STUDY TOGGLE ---
            # To get Variant A (Baseline RLIP), use: out['rlip_logits']
            # To get Variant B (+ Prior DB), use:    out['prior_logits']
            # To get Full Model (25.95%), use:       out['final_hoi']
            scores_batch = torch.sigmoid(out['final_hoi']).cpu().numpy()  # (Num_Pairs, 600)
 
            # Map predictions back to the formatted list
            for i, (h_box, o_box, candidate_hoi_ids, det_conf) in enumerate(pairs):
                scores = scores_batch[i]
                for hoi_id in candidate_hoi_ids:
                    cls_idx = hoi_id - 1
                    predictions.append({
                        'image_id': image_id,
                        'human_box': list(h_box),
                        'object_box': list(o_box),
                        'hoi_id': hoi_id,
                        'score': float(scores[cls_idx]) * float(det_conf),
                    })
 
    return predictions


def load_hico_gt_annotations(json_path):
    """Parses the standard HICO-DET test annotation JSON."""
    with open(json_path, 'r') as f:
        entries = json.load(f)
 
    ground_truths = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        image_id = entry['file_name']
        boxes = entry['annotations']  
        for hoi in entry['hoi_annotation']:
            h_box = boxes[hoi['subject_id']]['bbox']
            o_box = boxes[hoi['object_id']]['bbox']
            hoi_id = hoi['category_id']
            ground_truths[hoi_id][image_id].append((h_box, o_box))
 
    return ground_truths
 
 
def evaluate_final_test_yolo(model, image_items, yolo_model, device, hoi_list,
                              hoi_to_obj, preprocess_pair, gt_json_path,
                              det_score_thresh=0.3, num_hoi=600):
    predictions = run_inference_yolo(
        model, image_items, yolo_model, device, hoi_to_obj,
        preprocess_pair, det_score_thresh, num_hoi
    )
    ground_truths = load_hico_gt_annotations(gt_json_path)
    rare_ids = _load_rare_ids()
    return evaluate_hico_official(predictions, ground_truths, hoi_list, rare_ids, num_hoi)
