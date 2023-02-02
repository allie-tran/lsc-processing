
import bamboolib as bam
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm
tqdm.pandas()

from gps_utils import *
from vision_utils import *

# Parameters
MIN_PTS = 3
##############
# CLUSTERING #
##############
# %%
gps = pd.read_csv('files/cleaned_gps.csv')
minute_id_to_index = {minute_id:i for (i, minute_id) in enumerate(gps['minute_id'].values.tolist())}
MONTHS = [f"2019{i:0>2}" for i in range(1, 13)] + [f"2020{i:0>2}" for i in range(1, 7)]
EPS=0.05/6371 #0.01 = 10 meters
# visual_file = '../../original_data/lsc22_visual_concepts.csv'
visual_file = "files/visual_with_movement.csv"

# %%
cluster_to_stop = []
DAYS = [f"{i+1:0>2}" for i in range(31)]
for MONTH in tqdm(MONTHS, desc="Clustering"):
    for DAY in DAYS:
        # Filter by month
        gps_drop = gps.loc[gps['minute_id'].str.startswith(MONTH + DAY, na=False)]
        if len(gps_drop) > 0:
            total = len(gps_drop)
            gps_drop = gps_drop.dropna(subset=['latitude'])
            if len(gps_drop) > 0:
                # Clustering
                clustering = DBSCAN(eps=EPS, min_samples=MIN_PTS, algorithm='ball_tree', metric='haversine')
                clustering.fit(np.radians(gps_drop[['latitude', 'longitude']]))
                gps_drop['cluster_label'] = clustering.labels_
                gps_drop['core'] = [i in clustering.core_sample_indices_ for i in range(len(gps_drop))]
                # Assign cluster labels
                # gps_drop = gps_drop.loc[~(gps_drop['cluster_label'] == -1)]
                gps_drop = gps_drop.loc[gps_drop['core'] == True]
                # print(f"Outliers from {MONTH + DAY}:", clustering.labels_[clustering.labels_ == -1].size, ". Cores:", len(gps_drop), "Original:", total)
                if len(gps_drop):
                    clusters = gps_drop.groupby((gps_drop['cluster_label'].shift() != gps_drop['cluster_label']).cumsum()).agg(
                                                                    time_duration=('minute_id', time_duration),
                                                                    start=('minute_id', 'first'),
                                                                    end=('minute_id', 'last'),
                                                                    mean_spead=('speed', 'mean'),
                                                                    minute_id=('minute_id', list_all),
                                                                    label=('cluster_label', 'first'))
                    # Merge data points together based on their clusters
                    clusters = clusters.reset_index()
                    # clusters = clusters.loc[~(clusters['time_duration'] < 3)]
                    # Classify intial data points as stop/move
                    all_names = []
                    for index, row in clusters.iterrows():
                        if row["label"] != -1:
                            cluster_to_stop.append((row["start"], row['end'], row['minute_id'], f'{DAY}_{row["label"]}'))

# %% [markdown]
# ## Classify intial data points as stop/move

# %%
def blank_column(value, length):
    return [value for i in range(length)]

stop_values = blank_column(False, len(gps))
cluster_label_values = blank_column("", len(gps))

for start, end, minute_ids, cluster in tqdm(cluster_to_stop, desc="Classify intial data points"):
    start_id = minute_id_to_index[start]
    end_id = minute_id_to_index[end]
    for minute_id in range(start_id, end_id + 1):
        stop_values[minute_id] = True
        cluster_label_values[minute_id] = cluster

gps = gps.assign(stop=stop_values,
                 cluster_label=cluster_label_values)

# %% [markdown]
# # Post Processing

# %% [markdown]
# ## Smoothing

# %%

# %%
# Smooth stop/move label
gps["stop"] = smooth(gps["stop"])
gps["cluster_label"] = smooth(gps["cluster_label"])
gps["stop_label"] = gps.progress_apply(lambda x: x['cluster_label'] if x['stop'] else "", axis=1)

# %% [markdown]
# ## Change to ImageID index and recalculate movement

# %%
# Split images into different rows:
gps["ImageID"]=gps["ImageID"].str.split(",")
gps = gps.explode("ImageID").reset_index()
gps["ImageID"] = gps["ImageID"].str.replace(r'(\[|\]|\'|\s)', '', regex=True)
gps.loc[gps['ImageID'] == "", 'ImageID'] = np.nan
gps = gps.loc[gps['ImageID'].notna()]
gps = gps.drop(columns=['movement'])

# Merge into one file with the visual concepts
visual = pd.read_csv(visual_file, sep=',', decimal='.')
visual = visual.drop(columns=['stop', 'stop_label', 'Unnamed: 0.1', 'Unnamed: 0'])
# start from here
both = pd.merge(
    visual,
    gps,
    how="right",
    on='ImageID',
)
both = both.drop(columns=['Unnamed: 0', 'index'])

# # %%
# # Assign movement
# both["movement"] = [None for i in range(len(both))]
# both["movement_prob"] = [0 for i in range(len(both))]
# for i, row in tqdm(both.iterrows(), total=len(both), desc='assign movement'):
#     image_features = get_stop_embeddings([row["ImageID"]])
#     try:
#         image_features = torch.tensor(image_features).cuda().float()
#     except RuntimeError as e:
#         continue
#     movement, prob = movement_mode(list(moves.keys()), image_features)
#     both.loc[i, "movement"] = moves[movement]
#     both.loc[i, "movement_prob"] = prob

# # %%
theta = 0.9
# both["movement"] = smooth(both["movement"], 3)
# both["inside"] = both["movement"] == "Inside"
both.loc[(both["inside"] == False) & (both["movement_prob"] > theta), 'stop'] = False
both.loc[(both["inside"] == False) & (both["movement_prob"] > theta), 'stop_label'] = ""


# %% [markdown]
# ## Remove short stops

# %%
stops = both.groupby(((both['stop_label'].shift() != both['stop_label']) | (both['stop'].shift() != both['stop'])).cumsum()).agg(
                                                          inside=('inside', 'first'),
                                                          lat=('original_lat', 'mean'),
                                                          lon=('original_lng', 'mean'),
                                                          all_lon=('original_lng', list_all),
                                                          all_lat=('original_lat', list_all),
                                                          images=('ImageID', list_all),
                                                          stop=('stop', 'first'),
                                                          stop_label2=('stop_label', most_common),
                                                          movement=('movement', most_common),
                                                          duration=('ImageID', image_time_duration))
stops = stops.reset_index()
# stops = stops.drop(columns=['stop_label'])
stops = stops.rename(columns={'stop_label2': 'stop_label'})

# %%
stops = stops[['stop', 'movement'] + ['lat', 'lon', 'stop_label', 'all_lat', 'all_lon', 'images', 'duration']]
stops.loc[stops['duration'] < 3, 'stop'] = False

# %% [markdown]
# ## Adjusting boundaries

# %%
all_image_ids = list(both["ImageID"])
stop_values = [False] * len(both)
cluster_label_values = [""] * len(both)
boundaries = [None] * len(both)

for i, row in tqdm(stops.iterrows(), total=len(stops), desc='adjusting boundaries'):
    if row["stop"]:
        start = row["images"][0]
        boundaries[all_image_ids.index(start)] = "start"
        end = row["images"][-1]
        boundaries[all_image_ids.index(end)] = "end"

    for image in row["images"]:
        image_id = all_image_ids.index(image)
        stop_values[image_id] = row["stop"]
        if row["stop"]:
            cluster_label_values[image_id] = row["stop_label"]

both = both.assign(stop=stop_values,
                   stop_label=cluster_label_values,
                   boundary=boundaries)

# %%
stop_values = list(both["stop"])
cluster_label_values = list(both["stop_label"])
movements = list(both["movement"])
boundaries = list(both["boundary"])

# Forward
for i in tqdm(range(1, len(both)), desc="Verifying forward"):
    if boundaries[i] == "end": #Considering a stop ending boundaries
        # Checking if the cluster should end later (still inside)
        j = i + 1
        while j < len(both) and movements[j] == "Inside" and cluster_label_values[j] == "":
            stop_values[j] = True
            cluster_label_values[j] = cluster_label_values[i]
            j +=1

        # Checking if the cluster should end earlier (not inside anymore)
        j = i
        while j > 0 and movements[j] != "Inside" and cluster_label_values[j] == cluster_label_values[i]:
            stop_values[j] = False
            cluster_label_values[j] = ""
            j -= 1


# Backward
for i in tqdm(range(1, len(both)), desc="Verifying backward"):
    if boundaries[-i] == "start": #Considering a stop ending boundaries
        # Checking if the cluster should start sooner (going inside already)
        j = i + 1
        while j < len(both) and movements[-j] == "Inside" and cluster_label_values[-j] == "":
            stop_values[-j] = True
            cluster_label_values[-j] = cluster_label_values[-i]
            j += 1

        # Checking if the cluster should start later (not inside yet)
        j = i
        while j > 0 and movements[-j] != "Inside" and cluster_label_values[j] == cluster_label_values[i]:
            stop_values[j] = False
            cluster_label_values[j] = ""
            j -= 1

both = both.assign(stop=stop_values,
                 stop_label=cluster_label_values)

# %%
# # Smooth stop/move label
# both["stop"] = smooth(both["stop"])
# both["stop_label"] = smooth(both["stop_label"])
both.loc[(both["stop_label"] == "") & (both["inside"] == True), "stop_label"] = "INSIDE"

# %% [markdown]
# # Final stop/move

# %%
stops = both.groupby(((both['stop_label'].shift() != both['stop_label'])).cumsum()).agg(
                                                          inside=('inside', 'first'),
                                                          lat=('original_lat', 'mean'),
                                                          lon=('original_lng', 'mean'),
                                                          all_lon=('original_lng', list_all),
                                                          all_lat=('original_lat', list_all),
                                                          images=('ImageID', list_all),
                                                          stop=('stop', 'first'),
                                                          stop_label2=('stop_label', most_common),
                                                          movement=('movement', most_common),
                                                          duration=('ImageID', image_time_duration))
stops = stops.reset_index()
stops = stops.drop(columns=['stop_label'])
stops = stops.rename(columns={'stop_label2': 'stop_label'})

# %%
def calculate_distance(all_lat, all_lon, lat, lon):
    dists = [distance(lt, ln, lat, lon) for (lt, ln) in zip(all_lat, all_lon)]
    dists = [d for d in dists if d]
    if dists:
        return max(dists)
    return 50

stops["max_radius"] = stops.progress_apply(lambda x: calculate_distance(x['all_lat'],
                                                               x['all_lon'],
                                                               x['lat'],
                                                               x['lon']), axis=1)

# %%
stops = stops[['inside', 'stop', 'movement', 'stop_label'] + ['lat', 'lon', 'all_lat', 'all_lon','max_radius', 'images', 'duration']]
stops = stops.reset_index()

# %% [markdown]
# # Verification

# %%
theta = 0.9
def verification(row, logging=False):
    stop = stops.loc[row, "stop"]
    lat = stops.loc[row, "lat"]
    lon = stops.loc[row, "lon"]
    max_radius = stops.loc[row, "max_radius"]
    images = stops.loc[row, "images"]
    image_features = None
    if isinstance(images, str):
        images = json.loads(images.replace("'", '"'))
    image_features = get_stop_embeddings(images)

    # Get transport move
    if not image_features is None:
        image_features = torch.tensor(image_features).cuda().float()
        movement, movement_prob = movement_mode(list(moves.keys()), image_features)
        movement = moves[movement]
        if logging:
            print("Movement:", movement, movement_prob)
        if movement_prob > theta and movement in ["Inside", "Airport"]:
            stop = True
        elif movement_prob > theta:
            stop = False
        elif max_radius < 100 and movement in ["Inside", "Airport"]: # Low probability but small distance
            stop = True

        stops.loc[row, "stop"] = stop
        if not stop:
            stops.loc[row, "movement"] = movement
            stops.loc[row, "movement_prob"] = movement_prob

num = len(stops)
# num = 20
for i in tqdm(range(num), desc="Verification"):
    verification(i)

# %%
stops = stops[['inside', 'stop', 'movement', 'stop_label'] + ['lat', 'lon', 'max_radius', 'all_lat', 'all_lon', 'images', 'duration']]
stops = stops.reset_index()

# %% [markdown]
# # Save

# %%
stops.to_csv("files/stops.csv")
both.to_csv("files/image_with_gps.csv")
