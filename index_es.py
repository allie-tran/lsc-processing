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

info_dict = json.load(open("files/info_dict.json"))
# Set up ElasticSearch

es = Elasticsearch("http://localhost:9200")
interest_index = "lsc2023"
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
                "sort.field": ["timestamp"],
                "sort.order": ["asc"]
            },
            "mappings": {
                "properties": {
                    "image_path": {
                        "type": "keyword"
                    },
                    "descriptions": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "weekday": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "day": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "month": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "year": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "date": {
                        "type": "keyword", "similarity": "boolean"
                    },
                    "day_month": {"type": "keyword", "similarity": "boolean"
                    },
                    "month_year": {"type": "keyword", "similarity": "boolean"},
                    "day_year": {"type": "keyword", "similarity": "boolean"},
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
                    "time": {
                        "type": "date",
                        "format": "yyyy/MM/dd HH:mm:00Z"
                    },
                    "utc_time": {
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
                    "seconds_from_midnight": {"type": "long"},
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

def index(items):
    requests = []
    no_clip = 0
    for (image, desc) in tqdm(items):
        if es.exists(index=interest_index, id=image):
            continue
        if image in clip_embeddings:
            desc["clip_vector"] = clip_embeddings[image]
        else:
            no_clip += 1
            continue
        # if image.split('/')[-1] in similar_features:
            # desc["similar_vector"] = similar_features[image.split('/')[-1]]
        # else:
            # no_clip += 1
            # continue

        # datetime_value = datetime.strptime(desc["time"], "%Y/%m/%d %H:%M:00%z")
        desc["date"] = desc["time"][:10]
            
        desc["day"] = desc["time"][8:10]
        desc["month"] = desc["time"][5:7]
        desc["year"] = desc["time"][:4]
        
        desc["day_year"] = desc["day"] + "/" + desc["year"]
        desc["month_year"] = desc["month"] + "/" + desc["year"]
        desc["day_month"] = desc["day"] + "/" + desc["month"]
        
        desc["minute"] = int(desc["time"][14:16])
        desc["hour"]=int(desc["time"][11:13])

        if sys.getsizeof(requests) + sys.getsizeof(desc) > 15000:
            try:
                es.bulk(body=requests)
            except Exception as e:
                print(sys.getsizeof(requests), image)
                raise(e)
            requests = []

        requests.append(
            {"index": {"_index": interest_index, "_id": image}})
        requests.append(desc)

    if requests:
        es.bulk(body=requests)
    return no_clip

info_dict = list(info_dict.items())
# tasks = []
# num_process = 5
# num_items_per_task = 10000
# for i in range(0, len(info_dict), num_items_per_task):
    # tasks.append(info_dict[i: i+num_items_per_task])

# with mp.Pool(num_process) as pool:
    # no_clips = list(tqdm(pool.imap_unordered(index, tasks), total=len(tasks)))
no_clip = index(info_dict)
print(f"{no_clip} images were not indexed because there are no embeddings")
