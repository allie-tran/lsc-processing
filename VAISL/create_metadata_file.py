# %%
import bamboolib as bam
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
import multiprocess as mp

import numpy as np
tqdm.pandas()

# %% [markdown]
# # Read files

# %%
gps = pd.read_csv(r'files/cleaned_gps.csv', sep=',', decimal='.')
minute_id_to_index = {minute_id:i for (i, minute_id) in enumerate(gps['minute_id'].values.tolist())}

# %%
# Split images into different rows:
gps["ImageID"]=gps["ImageID"].str.split(",")
gps = gps.explode("ImageID").reset_index()
gps["ImageID"] = gps["ImageID"].str.replace(r'(\[|\]|\'|\s)', '', regex=True)
gps.loc[gps['ImageID'] == "", 'ImageID'] = np.nan
gps = gps.loc[gps['ImageID'].notna()]

# %%
# Merge into one file with the visual concepts
visual = pd.read_csv(r'../../original_data/lsc22_visual_concepts.csv', sep=',', decimal='.')
# start from here
both = pd.merge(
    visual,
    gps,
    how="right",
    on='ImageID',
)

all_minute_ids = both['minute_id'].values.tolist()
all_image_ids = both['ImageID'].values.tolist()
image_id_to_index = {image_id: i for (i, image_id) in enumerate(all_image_ids)}
minute_id_to_images = defaultdict(lambda:[])
for minute_id, image_id in zip(all_minute_ids, all_image_ids):
    minute_id_to_images[minute_id].append(image_id)

# %%
backup = both.copy()

# %%
both.to_csv("files/images.csv")

# %% [markdown]
# # VAISL results

# %%
stops = pd.read_csv(r'files/semantic_stops.csv', sep=',', decimal='.')

# %%
df = stops
# %%
import json
# cluster_to_name = [(metadata.index.values[:782].tolist(), "HOME", 53.38998, -6.1457602, True)]
cluster_to_name = []
all_images = set()

for index, row in df.iterrows():
    try:
        if row["first"] == "nan":
            continue
#         minute_ids = json.loads(row["minute_id"].replace("'", '"'))
#         cluster_to_name.append((minute_ids, row["checkin"], row["lat"], row["lon"], True))
        start = image_id_to_index[row["first"].strip('[], ').split('.')[0] + ".jpg"]
        end = image_id_to_index[row["last"].strip('[] ').split('.')[0] + ".jpg"]
        assert start <= end, "wrong order"
        image_ids = all_image_ids[start : end+1]
        if "movement" in row:
            cluster_to_name.append((image_ids, row["checkin"] if row["stop"] else row["movement"], row["lat"], row["lon"], row["stop"]))
        else:
            cluster_to_name.append((image_ids, row["checkin"], row["lat"], row["lon"], row["stop"]))
    except Exception as e:
        print(row)
        raise(e)

# %%
both = both.set_index("ImageID")

def classify(params):
    index, cluster = params
    image_ids, name, centre_lat, centre_lon, is_stop = cluster
    results = []
    for image_id in image_ids:
        if not is_stop or not np.isnan(both.loc[image_id, "latitude"]):
            centre_lat = both.loc[image_id, "latitude"]
            centre_lon = both.loc[image_id, "longitude"]
#         results.append([image_id, name] + [df.iloc[index][label] for label in ["found", "best_name_google", "best_label_google",
#                          "best_prob_google", "best_place_id_google", "cluster_label"]] + [centre_lat, centre_lon])
        results.append([image_id, name, centre_lat, centre_lon, is_stop])
    return results

with mp.Pool(mp.cpu_count()) as pool:
    results = list(tqdm(pool.imap_unordered(classify, enumerate(cluster_to_name)), total=len(cluster_to_name)))
results = [r for res in results for r in res]
len(results)

# %%
image_ids_all, new_names, lats, lons, is_stops = zip(*results)
both["new_lat"] = both["latitude"]
both["new_long"] = both["longitude"]
both["new_name"] = [None] * len(both)
both["stop"] = ["ERR"] * len(both)
def get_column(params):
    label, values = params
    row_to_name = {image_id_to_index[image_id]: name for image_id, name in zip(image_ids_all, values)}
    column = [row_to_name[i] if i in row_to_name else both.iloc[i][label] for i in range(len(both))]
    return column

with mp.Pool(mp.cpu_count()) as pool:
    rr =list(tqdm(pool.imap(get_column, [("new_name", new_names),
                                           ("new_lat", lats),
                                           ("new_long", lons),
                                           ("stop", is_stops)]), total=4))

# %%
rr[0] = ["HOME" for i in range(240)] + rr[0][240:]
rr[1] = [53.38998 for i in range(240)] + rr[1][240:]
rr[2] = [-6.1457602 for i in range(240)] + rr[2][240:]
rr[3] = [True for i in range(240)] + rr[3][240:]
both["new_name"] = rr[0]
both["new_lat"] = rr[1]
both["new_long"] = rr[2]
both["stop"] = rr[3]


# %%
both["new_lat"] = both["new_lat"].ffill()
both["new_long"] = both["new_long"].ffill()
both = both.reset_index()
both[['new_name']] = both[['new_name']].fillna('')

# %% [markdown]
# # Get city names

# %%
from map_apis import *
both['city'] = both.progress_apply(lambda x: get_cities(round(x['new_lat'], 3), round(x['new_long'], 3)), axis=1)
both.to_csv('files/final_metadata.csv')

# %%
