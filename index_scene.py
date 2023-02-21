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
import multiprocessing as mp

scene_dict = json.load(open("files/scene_dict.json"))
# Set up ElasticSearch

es = Elasticsearch([{"host": "localhost", "port": 9200}])
interest_index = "lsc2023_scene"
# clip_embeddings = joblib.load(
    # "/mnt/data/nvtu/embedding_features/L14_336_features_128.pkl")
photo_features = np.load("files/embeddings/features.npy")
photo_ids = list(pd.read_csv("files/embeddings/photo_ids.csv")["photo_id"])
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
                "sort.field": ["timestamp"],
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
                            "dims": 768,
                            "model": "permutation_lsh",         # 3
                            "k": 400,                            # 4
                            "repeating": True                   # 5
                        }
                    }
                }
            }
        }
    )


def index(items):
    requests = []
    no_clip = 0
    for (scene, desc) in tqdm(items):
        if es.exists(index=interest_index, id=scene):
            continue
        desc["scene"] = scene
        clip_vector = []
        for image in desc["images"]:
            if image in clip_embeddings:
                clip_vector.append(clip_embeddings[image])
        if clip_vector:
            desc["clip_vector"] = np.stack(clip_vector).mean(0)
            
            desc["date"] = desc["start_time"][8:10]
            desc["month"] = desc["start_time"][5:7]
            desc["year"] = desc["start_time"][:4]
            desc["minute"] = int(desc["start_time"][14:16])
            desc["hour"]=int(desc["start_time"][11:13])

            if sys.getsizeof(requests) + sys.getsizeof(desc) > 15000:
                try:
                    es.bulk(body=requests)
                except Exception as e:
                    print(sys.getsizeof(requests), scene)
                    raise(e)
                requests = []

            requests.append(
                {"index": {"_index": interest_index, "_id": scene}})
            requests.append(desc)

    if requests:
        es.bulk(body=requests)
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
