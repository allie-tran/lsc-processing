import json
import os
import sys
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.cluster import OPTICS
import multiprocessing as mp
import copy 
from numpy import linalg as LA

scene_dict = json.load(open("files/scene_dict.json"))
# Set up ElasticSearch

es = Elasticsearch("http://localhost:9200")
interest_index = "lsc2023_scene_mean"
# clip_embeddings = joblib.load(
    # "/mnt/data/nvtu/embedding_features/L14_336_features_128.pkl")
CLIP_EMBEDDINGS = os.environ.get("CLIP_EMBEDDINGS")
photo_features = np.load(f"{CLIP_EMBEDDINGS}/ViT-H-14_laion2b_s32b_b79k_nonorm/features.npy")
photo_ids = list(pd.read_csv(f"{CLIP_EMBEDDINGS}/ViT-H-14_laion2b_s32b_b79k_nonorm/photo_ids.csv")["photo_id"])
clip_embeddings = {photo_id: photo_feature for photo_id, photo_feature in zip(photo_ids, photo_features)}
photo_features = None
photo_ids = None

if es.indices.exists(index=interest_index):
    to_delete = input(
        f"Do you want to delete existing index: {interest_index}? (Y/N) ")
    if to_delete == "Y":
        print("Deleting index: " + interest_index)
        es.indices.delete(index=interest_index)

if not es.indices.exists(index=interest_index):
    es.indices.create(
        index=interest_index,
        **{
            "settings": {
                "number_of_shards": 8,
                "elastiknn": True,               # 2
                "number_of_replicas": 0,
                "sort.field": ["scene"],
                "sort.order": ["asc"]
            },
            "mappings": {
                "properties": {
                    "scene": {
                        "type": "keyword"
                    },
                    "images": {
                        "type": "keyword"
                    },
                    "descriptions": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "weekday": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "date": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "month": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "year": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "hour": {
                        "type": "byte"
                    },
                    "minute": {
                        "type": "byte"
                    },
                    "location": {
                        "type": "text"
                    },
                    "address": {
                        "type": "text"
                    },
                    "start_time": {
                        "type": "date",
                        "format": "yyyy/MM/dd HH:mm:00Z"
                    },
                    "end_time": {
                        "type": "date",
                        "format": "yyyy/MM/dd HH:mm:00Z"
                    },
                    "gps": {
                        "type": "geo_point"
                    },
                    "region": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "group": {"type": "keyword"},
                    "scene": {"type": "keyword"},
                    "timestamp": {"type": "long"},
                    "before": {"type": "keyword"},
                    "after": {"type": "keyword"},
                    "ocr": {"type": "text"},
                    "ocr_score": {
                        "type": "rank_features"
                    },
                    "clip_vector": {
                        "type": "elastiknn_dense_float_vector",
                        "elastiknn": {
                            "dims": 1024,
                            "model": "permutation_lsh",         # 3
                            "k": 400,                            # 4
                            "repeating": True                   # 5
                        }
                    }
                }
            }
        }
    )



METHOD = "MEAN"
# choose one of ["TRANS_WEIGHTS", "TRANS", "MEAN", "CLUSTERS"]
if "TRANS" in METHOD:
    from lifelog_qa.multiclip import load_scene_model, SequentialAgg
    import torch
    MODEL_VER = SequentialAgg
    save_file = "/home/tlduyen/LQA/MultiCLIP/models/SequentialAgg_2.pt"
    device = "cuda"
    train_config = {}
    if os.path.exists(save_file):
        states = torch.load(save_file, map_location="cpu")
        if "config" in states:
            train_config = states["config"]
    else:
        print("file not found!")
        exit()

    scene_agg, preprocess, *_ = load_scene_model(save_file, MODEL_VER, train_config,
                                            "cpu", train=False)
    scene_agg = scene_agg.to(device)
    scene_agg.eval()

cluster = OPTICS(min_samples=2, max_eps=0.5, metric='cosine')
def index(items):
    num_clusters = 0
    requests = []
    no_clip = 0
    pbar = tqdm(enumerate(items), total=len(items))
    for num_scene, (scene, desc) in pbar:
        if es.exists(index=interest_index, id=f"{scene}_0"):
            continue
        desc["scene"] = scene
        clip_vector = []
        
        for image in desc["images"]:
            if image in clip_embeddings:
                clip_vector.append(clip_embeddings[image])
        weights = None
        scene_embeddings = []
        if clip_vector:
            if METHOD == "TRANS_WEIGHTS":
                image_embeddings = torch.tensor(
                    clip_vector).to(device).float()
                with torch.no_grad():
                    (scene_embeddings, weights), _ = scene_agg.get_scene_embeddings(
                        image_embeddings, [2], None)
                    scene_embeddings = [scene_embeddings.squeeze(0).cpu().numpy()]
                    weights = weights.squeeze(0).cpu().numpy()
                cluster_images = [desc["images"]]
            elif METHOD == "TRANS":
                image_embeddings = torch.tensor(
                    clip_vector).to(device).float()
                with torch.no_grad():
                    (scene_embeddings, _), _ = scene_agg.get_scene_embeddings(
                        image_embeddings, [2], None)
                    scene_embeddings = [
                        scene_embeddings.squeeze(0).cpu().numpy()]
                cluster_images = [desc["images"]]
            else:
                scene_embeddings = [np.stack(clip_vector).mean(0)]
                cluster_images = [desc["images"]]
                if METHOD == "CLUSTERS":
                    frame_count = len(clip_vector)
                    if frame_count > 2:
                        cluster.fit(clip_vector)
                        labels = cluster.labels_
                        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                        for i in range(n_clusters_):
                            embedding = np.stack([vector for (vector, label) in zip(
                                clip_vector, labels) if label==i]).mean(0)
                            cluster_images.append([image for (image, label) in zip(
                                desc["images"], labels) if label==i])
                            scene_embeddings.append(embedding)
            
            desc["date"] = desc["start_time"][8:10]
            desc["month"] = desc["start_time"][5:7]
            desc["year"] = desc["start_time"][:4]
            desc["minute"] = int(desc["start_time"][14:16])
            desc["hour"]=int(desc["start_time"][11:13])
            desc["gps"] = []
            desc["weights"] = weights

            for i, (new_vector, label) in enumerate(zip(scene_embeddings, cluster_images)):
                num_clusters += 1
                new_desc = copy.deepcopy(desc)
                # new_desc['clip_vector'] = new_vector / LA.norm(new_vector, axis=-1, keepdims=True)
                new_desc['clip_vector'] = new_vector
                new_desc['cluster_images'] = label
                if sys.getsizeof(requests) + sys.getsizeof(new_desc) > 15000:
                    try:
                        es.bulk(body=requests)
                    except Exception as e:
                        print(sys.getsizeof(requests), scene)
                        raise(e)
                    requests = []
                requests.append(
                    {"index": {"_index": interest_index, "_id": f"{scene}_{i}"}})
                requests.append(new_desc)
        pbar.set_description(f"Num clusters: {num_clusters} ({num_clusters/(num_scene + 1):0.2f})")

    if requests:
        es.bulk(body=requests)
    print("Number of clusters:", num_clusters)
    return no_clip

scene_dict = list(scene_dict.items())
# tasks = []
# num_process = 5
# num_items_per_task = 10000
# for i in range(0, len(info_dict), num_items_per_task):
    # tasks.append(info_dict[i: i+num_items_per_task])

# with mp.Pool(num_process) as pool:
    # no_clips = list(tqdm(pool.imap_unordered(index, tasks), total=len(tasks)))
no_clip = index(scene_dict)
print(f"{no_clip} images were not indexed because there are no embeddings")
