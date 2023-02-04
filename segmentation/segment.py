import json
from collections import Counter, defaultdict
from tqdm import tqdm
import joblib
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cdist
import os
import cv2
import numpy as np
import pandas as pd
from extract_sift import *
import json
from pprint import pprint
from datetime import datetime

metadata = pd.read_csv('../VAISL/files/final_metadata.csv', sep=',', decimal='.')
unhelpful_images = json.load(open("../files/unhelpful_images.json"))
photo_features = np.load("../files/embeddings/features.npy")
photo_ids = list(pd.read_csv("../files/embeddings/photo_ids.csv")["photo_id"])
clip_embeddings = {photo_id: photo_feature for photo_id,
                   photo_feature in zip(photo_ids, photo_features)}
photo_features = None
photo_ids = None

if __name__ == "__main__":
    last_feat = np.zeros(768)
    last_location = "NONE"
    last_location_info = ""

    groups = defaultdict(lambda: {"scenes": [], "location": "", "location_info": ""})
    num_group = 0
    num_scene = 0
    images = []
    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        if isinstance(row['ImageID'], str):
            if row['ImageID'] not in unhelpful_images:
                key = f"{row['ImageID'][:6]}/{row['ImageID'][6:8]}/{row['ImageID']}"
                if key in clip_embeddings:
                    feat = clip_embeddings[key]
                else:
                    feat = np.zeros(768)
                location = row["checkin"]
                location_info = row["categories"] if row["stop"] else row["checkin"]
                # Different group
                if location != last_location or len(groups[f"G_{num_group}"]["scenes"]) >= 99:
                    if images:
                        num_scene += 1
                        groups[f"G_{num_group}"]["scenes"].append((f"S_{num_scene}", images))
                        images = []
                    if groups[f"G_{num_group}"]["scenes"]:
                        groups[f"G_{num_group}"]["location"] = last_location
                        groups[f"G_{num_group}"]["location_info"] = location_info
                        num_group += 1
                    last_location = location
                    last_location_info = location_info

                else: # Might be same scene or different scene
                    new_scene = cdist([last_feat], [feat], 'cosine')[0]
                    if new_scene > 0.2: # Different scene
                        if images:
                            num_scene += 1
                            groups[f"G_{num_group}"]["scenes"].append((f"S_{num_scene}", images))
                            images = []
                images.append(key)
                last_feat = feat
    if images:
        num_scene += 1
        groups[f"G_{num_group}"]["scenes"].append((f"S_{num_scene}", images))

    json.dump(groups, open("../files/group_segments.json", "w"))
