import bamboolib as bam
from tqdm.auto import tqdm
tqdm.pandas()
import matplotlib.pyplot as plt
from PIL import Image
from gps_utils import *
from map_apis import *
from vision_utils import *
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


# %% ALL CONFIGS
CLASSIFIED_IMAGES = True
MIN_PTS = 3


# %%
metadata_file = '../../original_data/lsc22_medatada.csv'
checkin_file = '../../original_data/checkins.json'

movement_file = "files/lsc22_metadata_with_movement.csv"
MONTHS = [f"2019{i:0>2}" for i in range(1, 13)] + [f"2020{i:0>2}" for i in range(1, 7)]
print("Processing months:", MONTHS)

# %%
if not CLASSIFIED_IMAGES:
    metadata = pd.read_csv(metadata_file, sep=',', decimal='.')
    if len(MONTHS) == 1:
        metadata = metadata.loc[metadata['minute_id'].str.startswith(MONTHS[0], na=False)]
    metadata = metadata.reset_index()
    # Assign movement
    metadata["movement"] = [None for i in range(len(metadata))]
    metadata["movement_prob"] = [None for i in range(len(metadata))]
    metadata["inside"] = [None for i in range(len(metadata))]
    metadata["inside_prob"] = [None for i in range(len(metadata))]
    for i, row in tqdm(metadata.iterrows(), desc='Classifying minute ids', total=len(metadata)):
        images = row["ImageID"]
        if isinstance(images, str):
            images = json.loads(images.replace("'", '"'))
            if images:
                image_features = get_stop_embeddings(images)
            try:
                image_features = torch.tensor(image_features).cuda().float()
            except RuntimeError as e:
                continue
            movement, prob = movement_mode(list(moves.keys()), image_features)
            metadata.loc[i, "movement"] = moves[movement]
            metadata.loc[i, "movement_prob"] = prob

            inside, prob = movement_mode(list(insides.keys()), image_features)
            metadata.loc[i, "inside"] = insides[inside]
            metadata.loc[i, "inside_prob"] = prob
    metadata = metadata.drop(columns=['Unnamed: 0', 'index'])
    metadata.to_csv(movement_file)
# %%
metadata_file = movement_file

# %% [markdown]
# # Data Cleaning

# %%
metadata = pd.read_csv(metadata_file, sep=',', decimal='.')
if len(MONTHS) == 1:
    metadata = metadata.loc[metadata['minute_id'].str.startswith(MONTHS[0], na=False)]
metadata = metadata.drop(columns=['Unnamed: 0'])
metadata = metadata.reset_index()

# %% [markdown]
# ## Speed visualisation

# %%
# Calculate speed for each data points
def calculate_speeds(lats, lngs):
    assert len(lats) == len(lngs), "Different lengths of lats and lngs"
    speeds = [0.0]
    i = 0
    j = i + 1
    while j < len(lats):
        dist = distance(lats[j], lngs[j], lats[i], lngs[i])
        if dist:
            time_dist = j-i
            speed = dist/time_dist/60
        else:
            speed = None
        speeds.append(speed)
        i = j
        j += 1
    return speeds

gps = metadata.copy()
gps.loc[0, "latitude"] = 53.38998
gps.loc[0, "longitude"] = -6.1457602
print("Calculating speed...")
speeds = calculate_speeds(gps["latitude"], gps["longitude"])
gps['speed'] = speeds

# %%
confidence = gps.loc[(gps['movement_prob'] >= 0)]
def q1(x):
    return x.quantile(0.25)

def q3(x):
    return x.quantile(0.75)

movements = confidence.groupby(['movement']).agg(minute_id_size=('minute_id', 'size'),
                                                 mean=('speed', 'mean'),
                                                 std=('speed', 'std'),
                                                 q1=('speed', q1),
                                                 q3=('speed', q3)).reset_index()
movements["limit"] = movements["q3"] * 2.5 - 1.5 * movements["q1"]
movements = movements.dropna(subset=['limit'])

# %%
limits = dict(zip(movements["movement"], movements["limit"]))
limits["Airplane"] = 300

# %%
confidence["movement"] = confidence["movement"].str.replace('Outside', 'Outdoor', regex=False)
confidence["movement"] = confidence["movement"].str.replace('Inside', 'Indoor', regex=False)

# %% [markdown]
# ## Removal

# %%
def remove_outliners_with_movements(lats, lngs, all_images, limits={}, ignore=[]):
    assert len(lats) == len(lngs), "Different lengths of lats and lngs"
    i = 0
    to_remove = []
    speeds = [0.0]
    next_valid = []
    activities = []
    movement_valid = []
    j = i + 1
    key = ""
    pbar = tqdm(total=len(lats))
    pbar.update(1)
    while j < len(lats):
        dist = distance(lats[j], lngs[j], lats[i], lngs[i])
        accept = False
        if dist:
            time_dist = j-i
            speed = dist/time_dist/60
            if j in ignore:
                accept = True
            else:
                if not limits:
                    accept = True
                else:
                    if not isinstance(all_images[j], str): # No images available. Ok to cut
                        accept = True
                    else:
                        # Get movement mode from i -> j
                        images = []
                        for row in range(i, j+1):
                            if isinstance(all_images[row], str):
                                row_images = json.loads(all_images[row].replace("'", '"'))
                                if row_images:
                                    images.extend(row_images)
                        if images:
                            image_features = get_stop_embeddings(images)
                            try:
                                image_features = torch.tensor(image_features).cuda().float()
                                movement, movement_prob = movement_mode(list(moves.keys()), image_features)
                                key = moves[movement]
                                if key == "Airplane":
                                    accept = True
                                elif key in limits and speed < limits[key]:
                                    accept = True
                            except RuntimeError as e:
                                accept = False
        else:
            speed = None
            if np.isnan(lats[i]):
                accept = True

        if accept:
            # Acceptable distance
            speeds.append(speed)
            next_valid.extend([j for i in range(j-i)])
            activities.extend([key for i in range(j-i)])
            i = j
        else:
            # Delete that point and recalculate
            to_remove.append(j)
            speeds.append(speed)
        j += 1
        pbar.update(1)

    pbar.close()
    next_valid.extend([j for i in range(j-i)])
    activities.extend([key for i in range(j-i)])
    return to_remove, speeds, next_valid, activities

# %%
# [PAPER] If two consecutive data points is greater than 130km per hours, delete (130km/h = 2160m/minute)
gps = metadata.copy()
gps.loc[0,"latitude"] = 53.38998
gps.loc[0, "longitude"] = -6.1457602

# %%
print("Removing outliers")
to_remove, speeds, next_valid, activities = remove_outliners_with_movements(gps["latitude"], gps["longitude"], gps["ImageID"], limits)

# %%
gps["original_lat"] = gps["latitude"]
gps["original_lng"] = gps["longitude"]
gps.loc[to_remove, "latitude"] = np.nan
gps.loc[to_remove, "longitude"] = np.nan
gps['speed'] = speeds
gps['next_valid'] = next_valid
gps["activities"] = activities

# %% [markdown]
# # Gap Treatment

# %%
empty = gps.copy()
empty["ff_longitude"] = empty["longitude"].ffill(limit=1)
empty["ff_latitude"] = empty["latitude"].ffill(limit=1)
empty["bf_longitude"] = empty["longitude"].bfill(limit=1)
empty["bf_latitude"] = empty["latitude"].bfill(limit=1)

empty["longitude"][~empty["longitude"].isna()] = "valid"
empty["longitude"][empty["longitude"].isna()] = "nan"
empty = empty.groupby((empty['longitude'].shift() != empty['longitude']).cumsum()).agg(
                                                          minute_id=('minute_id', list_all),
                                                          valid=('longitude', 'first'),
                                                          duration=('minute_id', 'count'),
                                                          lat1=('ff_latitude', 'first'),
                                                          lng1=('ff_longitude', 'first'),
                                                          lat2=('bf_latitude', 'last'),
                                                          lng2=('bf_longitude', 'last'),
                                                          first_index=('index', 'first'),
                                                          last_index=('index', 'last'),
                                                          images=('ImageID', unpack_str),
                                                          movement=('movement', most_common)
                                                        )
empty = empty.reset_index()
empty = empty.drop(columns=['longitude'])
empty = empty.loc[empty['valid'].isin(['nan'])]
empty["distance"] = empty.progress_apply(lambda x: distance(x['lat1'], x['lng1'], x['lat2'], x['lng2']), axis=1)

# %%
for i, gap in tqdm(empty.iterrows(), desc="Filling in empty rows", total=len(empty)):
#     if gap["duration"] > MIN_PTS and row["distance"] > 50:
#         to_remove.append(row["last_index"] + 1)
    if gap["movement"] == "Airplane":
        continue
    if gap["duration"] > MIN_PTS:
        interval = (gap["duration"] - 1) // (MIN_PTS + 1) + 1
        total_intervals = gap["duration"] / interval
        j = 0
        for i in range(gap["first_index"] + interval, gap["last_index"] + 1, interval):
            j += 1
            gps.loc[i, "latitude"] = gap["lat1"] + (gap["lat2"] - gap["lat1"]) * j / total_intervals
            gps.loc[i, "longitude"] = gap["lng1"] + (gap["lng2"] - gap["lng1"]) * j / total_intervals
    else:
        if gap["movement"] in ["Still", "Walking Inside"] and gap["distance"]/gap["duration"] < limits[gap["movement"]]:
            gps.loc[gap["first_index"]-1: gap["last_index"] + 1, "latitude"] = gps.loc[gap["first_index"]-1: gap["last_index"] + 1, "latitude"].interpolate()
            gps.loc[gap["first_index"]-1: gap["last_index"] + 1, "longitude"] = gps.loc[gap["first_index"]-1: gap["last_index"] + 1, "longitude"].interpolate()

cleaned_gps = gps[['minute_id', 'latitude', 'longitude', 'ImageID', 'speed', 'movement', 'time_zone', 'original_lat', 'original_lng']]
cleaned_gps.to_csv('files/cleaned_gps.csv')

