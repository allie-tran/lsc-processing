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
import json
from pprint import pprint
from datetime import datetime
from dateutil import tz
from numpy import linalg as LA

metadata = pd.read_csv('../VAISL/files/final_metadata.csv', sep=',', decimal='.')
metadata['checkin'] = metadata['checkin'].fillna("")
unhelpful_images = json.load(open("../files/unhelpful_images.json"))
CLIP_EMBEDDINGS = os.environ.get("CLIP_EMBEDDINGS")
photo_features = np.load(f"{CLIP_EMBEDDINGS}/features.npy")
photo_features /= LA.norm(photo_features, keepdims=True, axis=-1)
photo_ids = list(pd.read_csv(f"{CLIP_EMBEDDINGS}/photo_ids.csv")["photo_id"])
dim = photo_features[0].shape[-1]
clip_embeddings = {photo_id: photo_feature for photo_id,
                   photo_feature in zip(photo_ids, photo_features)}
photo_features = None
photo_ids = None
THRESHOLD = 0.3

metadata['checkin'] = metadata['checkin'].fillna("")
metadata["new_timezone"] = metadata["new_timezone"].ffill()
metadata['categories'] = metadata['categories'].fillna("")

def is_new_group(location, last_location, location_info, last_location_info, scenes):
    if last_location != location:
        return True
    if last_location_info != location_info:
        return True
    if location == "NONE":
        return len(scenes) > 99
    return False    

def to_full_key(image):
    return f"{image[:6]}/{image[6:8]}/{image}"

def to_local_time(utc_time, time_zone):
    return utc_time.astimezone(tz.gettz(time_zone))

if __name__ == "__main__":
    last_feat = np.zeros(dim)
    last_location = "HOME"
    last_location_info = ""
    last_day = ""
    
    average = []
    groups = defaultdict(lambda: {"scenes": [], "location": "", "location_info": ""})
    
    num_group = 0
    num_scene = 0
    images = []
    pbar = tqdm(metadata.iterrows(), total=len(metadata))
    for index, row in pbar:
        if isinstance(row['ImageID'], str):
            if row['ImageID'] not in unhelpful_images:
                key = to_full_key(row['ImageID'])
                if key in clip_embeddings:
                    feat = clip_embeddings[key]
                else:
                    continue
                location = row["checkin"] if row["stop"] else "NONE"
                location_info = row["categories"] if row["stop"] else row["checkin"]
                utc_time = datetime.strptime(row["minute_id"]+"00", "%Y%m%d_%H%M%S").replace(tzinfo=tz.gettz('UTC'))
                local_time = to_local_time(utc_time, row["new_timezone"])
                day = datetime.strftime(local_time, "%A")
                # Different group
                if is_new_group(location, last_location, location_info, last_location_info, groups[f"G_{num_group}"]["scenes"]):
                    if images:
                        num_scene += 1
                        groups[f"G_{num_group}"]["scenes"].append((f"S_{num_scene}", images))
                        images = []
                    if groups[f"G_{num_group}"]["scenes"]:
                        groups[f"G_{num_group}"]["location"] = last_location
                        groups[f"G_{num_group}"]["location_info"] = last_location_info
                        num_group += 1
                    last_location = location
                    last_location_info = location_info
                else: # Might be same scene or different scene
                    # new_scene = cdist([last_feat], [feat], 'cosine')[0]
                    sim = last_feat @ feat.T 
                    average.append(sim)
                    if day != last_day or sim < (1 - THRESHOLD): #Different scene
                        if images:
                            num_scene += 1
                            groups[f"G_{num_group}"]["scenes"].append((f"S_{num_scene}", images))
                            images = []
                images.append(key)
                last_feat = feat
                last_day = day
        extra = ""
        # extra = f", 75% dist: {pd.Series(average).quantile(0.75)}"
        pbar.set_description(f"Scenes: {num_scene}({num_scene/(index+1):0.2f}), Groups: {num_group} {extra}")
    if images:
        num_scene += 1
        groups[f"G_{num_group}"]["scenes"].append((f"S_{num_scene}", images))
    print("Number of groups:", num_group)
    print("Number of scenes:", num_scene)
    json.dump(groups, open("../files/group_segments.json", "w"))
