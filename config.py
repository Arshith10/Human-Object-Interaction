import os
import json
import torch
import numpy as np
import random
from datetime import datetime

# Seed function
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

# Using relative paths for the H200 server
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "hico_root":  os.path.join(BASE_DIR, "data", "hico_20160224_det"),
    "output_dir": os.path.join(BASE_DIR, "data", "hico_det_preprocessed"),
    "prior_db_path": os.path.join(BASE_DIR, "data", "interaction_prior_database_qwen.pt"),
    "image_size": 224,
    "clip_mean": [0.48145466, 0.4578275,  0.40821073],
    "clip_std":  [0.26862954, 0.26130258, 0.27577711],
    "min_bbox_area": 100,
    "rare_threshold": 10,
}

for d in ["images/train", "images/test", "annotations", "stats"]:
    os.makedirs(os.path.join(CONFIG["output_dir"], d), exist_ok=True)
