import pandas as pd
import json
import multiprocess as mp
from tqdm.auto import tqdm
import numpy as np

tqdm.pandas()
from datetime import datetime
from gps_utils import list_all, unpack_str, image_time_duration

def image_to_date(image_id):
    return datetime.strptime(image_id, "%Y%m%d_%H%M%S_000.jpg")

def image_time_duration(series):
    series = list(series)
    return (image_to_date(series[-1]) - image_to_date(series[0])).seconds/60 + 1

def assign_to_images(df):
    both = pd.read_csv(r'files/image_with_gps.csv', sep=',', decimal='.')
    all_image_ids = both['ImageID'].values.tolist()
    image_id_to_index = {image_id: i for (i, image_id) in enumerate(all_image_ids)}

    def classify(params):
        index, cluster = params
        image_ids, name, centre_lat, centre_lon, is_stop, category, parent = cluster
        results = []
        for image_id in image_ids:
            if not is_stop or not np.isnan(both.loc[image_id, "latitude"]):
                if not np.isnan(both.loc[image_id, "original_lat"]): # This is NOT from gap filling
                    centre_lat = both.loc[image_id, "latitude"]
                    centre_lon = both.loc[image_id, "longitude"]
            results.append([image_id, name, centre_lat, centre_lon, is_stop, category, parent])
        return results

    cluster_to_name = []
    all_images = set()

    for index, row in df.iterrows():
        try:
            if row["first"] == "nan":
                continue
            start = image_id_to_index[row["first"].strip('[], ').split('.')[0] + ".jpg"]
            end = image_id_to_index[row["last"].strip('[] ').split('.')[0] + ".jpg"]
            assert start <= end, "wrong order"
            image_ids = all_image_ids[start : end+1]
            if "movement" in row:
                cluster_to_name.append((image_ids, row["checkin"] if row["stop"] else row["movement"], row["lat"], row["lon"], row["stop"], row["categories"], row["parent"]))
            else:
                cluster_to_name.append((image_ids, row["checkin"], row["lat"], row["lon"], row["stop"], row["categories"], row["parent"]))
        except Exception as e:
            print(row)
            raise(e)

    both = both.set_index("ImageID")
    results = []
    for params in tqdm(enumerate(cluster_to_name), total=len(cluster_to_name), desc='Classifying points'):
        results.append(classify(params))

    results = [r for res in results for r in res]
    image_ids_all, new_names, lats, lons, is_stops, categories, parents = zip(*results)
    both["new_lat"] = both["latitude"]
    both["new_long"] = both["longitude"]
    both["new_name"] = [""] * len(both)
    both["stop"] = ["ERR"] * len(both)
    both["categories"] = [""] * len(both)
    both["parent"] = [""] * len(both)

    def get_column(params):
        label, values = params
        row_to_name = {image_id_to_index[image_id]: name for image_id, name in zip(image_ids_all, values)}
        column = [row_to_name[i] if i in row_to_name else both.iloc[i][label] for i in range(len(both))]
        return column

    with mp.Pool(mp.cpu_count()) as pool:
        rr =list(tqdm(pool.imap(get_column, [("new_name", new_names),
                                               ("new_lat", lats),
                                               ("new_long", lons),
                                               ("stop", is_stops),
                                              ("categories", categories),
                                              ("parent", parents)]), 
                     total=6,
                     desc="Getting new values for columns"))

    rr[0] = ["HOME" for i in range(240)] + rr[0][240:]
    rr[1] = [53.38998 for i in range(240)] + rr[1][240:]
    rr[2] = [-6.1457602 for i in range(240)] + rr[2][240:]
    rr[3] = [True for i in range(240)] + rr[3][240:]
    both["new_name"] = rr[0]
    both["new_lat"] = rr[1]
    both["new_long"] = rr[2]
    both["stop"] = rr[3]
    both["categories"] = rr[4]
    both["parent"] = rr[5]

    both["new_lat"] = both["new_lat"].ffill()
    both["new_long"] = both["new_long"].ffill()
    both = both.reset_index()
    both[['new_name']] = both[['new_name']].fillna('')
    return both

def agg_stop(df):
    both = assign_to_images(df)

    stops = both.groupby((both['new_name'].shift() != both['new_name']).cumsum()).agg(
                                                              stop=('stop', 'first'),
                                                              checkin=('new_name', 'first'),
                                                              categories=('categories', 'first'),
                                                              parent=('parent', 'first'),
                                                              lat=('new_lat', 'mean'),
                                                              lon=('new_long', 'mean'),
                                                              all_lon=('new_long', list_all),
                                                              all_lat=('new_lat', list_all),
                                                              first=('ImageID', 'first'),
                                                              last=('ImageID', 'last'),
                                                              images=('ImageID', list_all),
                                                              duration=('ImageID', image_time_duration))
    stops = stops.reset_index()
    return stops
