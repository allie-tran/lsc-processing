# %%
from tqdm.auto import tqdm
import bamboolib as bam
import pandas as pd
tqdm.pandas()
from gps_utils import *
from map_apis import *
from vision_utils import *

# %%
from datetime import datetime, timezone, timedelta
# Language libraries
from unidecode import unidecode
import re

# %% [markdown]
# # Read Files

# %%
checkin_file = '../../original_data/checkins.json'
stop_file = 'files/stops.csv'
DISTANCE_THRESHOLD = 100
# %%
stops = pd.read_csv(stop_file, sep=',', decimal='.')
# Leave out stops without images
# stops['images'] = stops['images'].astype('string')
# stops = stops.loc[~(stops['images'].isin(['[]']))]
# stops = stops.loc[~(stops['lat'].isna())]
stops = stops.reset_index()
stops["first"] = [str(images).split(',')[0].strip("'[] ") for images in stops["images"]]
stops["last"] = [str(images).split(',')[-1].strip("'[] ") for images in stops["images"]]
EMPTY_STRINGS = ["" for i in range(len(stops))]
ZEROS = [0 for i in range(len(stops))]
FALSES = [False for i in range(len(stops))]

stops = stops.assign(checkin=EMPTY_STRINGS,
                     checkin_id=EMPTY_STRINGS,
                     in_checkin=FALSES,
                     original_name=EMPTY_STRINGS,
                     categories=EMPTY_STRINGS,
                     prob=ZEROS,
                     parent=EMPTY_STRINGS,
                     parent_id=EMPTY_STRINGS)
                     # movement=EMPTY_STRINGS,
                     # movement_prob=ZEROS)

# %%
checkins = json.load(open(checkin_file))
named_checkins = [{
                   "name": checkin["venue"]["name"],
                   "place_id": checkin["venue"]["id"],
                   "latitude": checkin["venue"]["latitude"],
                   "longitude": checkin["venue"]["longitude"],
                   "regions": [checkin["venue"][key] for key in ["city", "state", "country"] if key in checkin["venue"]],
                   "time": datetime.fromtimestamp(checkin["createdAt"]), #not sure
                   "timeZoneOffset": checkin["timeZoneOffset"]} for checkin in checkins][::-1]
# Filter time
named_checkins = [checkin for checkin in named_checkins if checkin["time"].year in [2019, 2020]]
NULL_CHECKIN = {"name": "Unknown Place", "place_id": "No Places Found", "categories": [], "parent": "", "parent_id": ""}

# %% [markdown]
# ## Enrich checkins

# %%
for checkin in tqdm(named_checkins):
    categories_res = get_categories(checkin['place_id'])
    checkin["categories"] = [cat['name'] for cat in categories_res["categories"]]
    if "description" in categories_res:
        checkin["description"] = categories_res["description"]
    else:
        checkin["description"] = ""
    if "related_places" in categories_res and "parent" in categories_res["related_places"]:
        checkin["parent"] = categories_res["related_places"]["parent"]["name"]
        checkin["parent_id"] = categories_res["related_places"]["parent"]["fsq_id"]
    else:
        checkin["parent"] = ""
        checkin["parent_id"] = ""
    if "related_places" in categories_res and "children" in categories_res["related_places"]:
        checkin["children"] = []
        for children in categories_res["related_places"]["children"][:20]:
            if "fsq_id" in children:
                children_res = get_categories(children["fsq_id"])
                if "description" in children_res:
                    children["description"] = children_res["description"]
                children["categories"] = [cat['name'] for cat in children_res["categories"]]
                checkin["children"].append(children)
    else:
        checkin["children"] = []

    photos_res = get_photos(checkin["place_id"])
    checkin["indoor_photos"] = [photo["prefix"] + "original" + photo["suffix"] for photo in photos_res if 'indoor' in photo["classifications"]]
    checkin["outdoor_photos"] = [photo["prefix"] + "original" + photo["suffix"] for photo in photos_res if 'outdoor' in photo["classifications"]]

# %% [markdown]
# ## Various checkin functions



# %% [markdown]
# ## Assign checkins

# %%
def change_airport_parents(possible_checkins, weights, existed_place_ids):
    # Change parents for airports:
    for checkin in possible_checkins:
        if detect_airport(checkin):
            categories_res = get_categories(checkin["parent_id"])
            categories = [cat['name'] for cat in categories_res["categories"]]
            parent = checkin["parent"]
            parent_id = checkin["parent_id"]

            if parent:
                break_threshold = 5
                i_to_break = 0
                while True:
                    i_to_break += 1
                    if i_to_break < break_threshold and "related_places" in categories_res and "parent" in categories_res["related_places"]:
                        parent = categories_res["related_places"]["parent"]["name"]
                        parent_id = categories_res["related_places"]["parent"]["fsq_id"]
                        categories_res = get_categories(parent_id)
                        categories = [cat['name'] for cat in categories_res["categories"]]
                    else:
                        break
                checkin["parent"] = parent
                checkin["parent_id"] = parent_id

            if parent_id in existed_place_ids:
                weights[existed_place_ids.index(parent_id)] += 1
    return possible_checkins, weights

# %%
def get_nearby_parents(nearbys, possible_checkins, weights, existed_place_ids):
    for i, checkin in enumerate(nearbys):
        checkin = parse_checkin(checkin)
        if detect_airport(checkin):
            categories_res = get_categories(checkin["parent_id"])
            categories = [cat['name'] for cat in categories_res["categories"]]
            parent = checkin["parent"]
            parent_id = checkin["parent_id"]
            if parent:
                break_threshold = 5
                i_to_break = 0
                while True:
                    i_to_break += 1
                    if i_to_break < break_threshold and "related_places" in categories_res and "parent" in categories_res["related_places"]:
                        parent = categories_res["related_places"]["parent"]["name"]
                        parent_id = categories_res["related_places"]["parent"]["fsq_id"]
                        categories_res = get_categories(parent_id)
                        categories = [cat['name'] for cat in categories_res["categories"]]
                    else:
                        break
                checkin["parent"] = parent
                checkin["parent_id"] = parent_id

        if checkin["place_id"] not in existed_place_ids:
            existed_place_ids.append(checkin["place_id"])
            possible_checkins.append(checkin)
            weights.append(1) #TODO!
        # else:
            # weights[existed_place_ids.index(checkin["place_id"])] += 1

    # Add parent to the list
    for checkin in possible_checkins:
        if checkin["parent_id"] and checkin["parent_id"] not in existed_place_ids:
            if checkin["parent"] != checkin["name"]:
                categories_res = get_categories(checkin["parent_id"])
                new_checkin = {"name": checkin["parent"],
                               "place_id": checkin["parent_id"],
                               "categories": [cat['name'] for cat in categories_res["categories"]],
                               "parent": "",
                               "parent_id": ""}
                if "description" in categories_res:
                    new_checkin["description"] = categories_res["description"]
                if "related_places" in categories_res and "parent" in categories_res["related_places"]:
                    new_checkin["parent"] = categories_res["related_places"]["parent"]["name"]
                    new_checkin["parent_id"] = categories_res["related_places"]["parent"]["fsq_id"]
                existed_place_ids.append(checkin["parent_id"])
                possible_checkins.append(new_checkin)
                weights.append(1)
    return possible_checkins, weights, existed_place_ids

# %% [markdown]
# ## Fill in rows

# %%
def detect_airport(checkin):
    if "airport" in ", ".join(checkin["categories"]).lower() or "airport" in checkin["name"].lower():
        return True
    return "airport" in checkin["parent"].lower()


# %%
checkin_times = [checkin["time"] for checkin in named_checkins]

def get_nearby_checkins(start, end, lat=None, lon=None, max_radius=500, time_limit=timedelta(hours=1)):
    start_ind = find_closest_index(checkin_times, start)
    end_ind = find_closest_index(checkin_times, end + time_limit)
    base_checkins, gaps = [], []
    for i in range(start_ind, end_ind+1):
        checkin = named_checkins[i]
        if lat:
            if distance(lat, lon, checkin["latitude"], checkin["longitude"]) > max(DISTANCE_THRESHOLD * 4, max_radius):
                continue
        base_checkins.append(checkin)
        gaps.append(checkin_times[i] - start)
    weights = [1 for i in range(len(base_checkins))]
    children_checkins = []
    for checkin, gap in zip(base_checkins, gaps):
        if not detect_airport(checkin):
            for children in checkin["children"]:
                children["place_id"] = children["fsq_id"]
                children["parent"] = checkin["name"]
                children["parent_id"] = checkin["place_id"]
                children_checkins.append(children)
                weights.append(1)
    return base_checkins + children_checkins, weights

# %%
def image_to_date(image_id):
    return datetime.strptime(image_id, "%Y%m%d_%H%M%S_000.jpg")

# %%
def find_named_checkins_nearby(images, image_features, stop, lat, lon, max_radius, logging=False):
    start = image_to_date(images[0])
    end = image_to_date(images[-1])
    if stop:
        possible_checkins, weights, existed_place_ids = [], [], []
        num_nearby_checkins = 0

        # Use checkins
        possible_checkins, weights = get_nearby_checkins(start, end, lat, lon, max_radius)
        existed_place_ids = [checkin["place_id"] for checkin in possible_checkins]
        num_nearby_checkins = len(possible_checkins)
        possible_checkins, weights = change_airport_parents(possible_checkins, weights, existed_place_ids)

        nearbys = get_nearby_places(round(lat, 3), round(lon, 3))["results"]
        if logging:
            print(images)
            print([(checkin["name"], checkin["distance"]) for checkin in nearbys])
            print("Max radius:", max_radius)
        # Filter nearbys by distance
        nearbys = [checkin for checkin in nearbys if "distance" not in checkin or checkin["distance"] < max(DISTANCE_THRESHOLD * 4, max_radius)]

        if not nearbys:
            if logging:
                print(images)
                print([checkin["name"] for checkin in get_nearby_places(round(lat, 3), round(lon, 3))["results"]])
                print("Max radius:", max_radius)
        else:
            # Get parents for nearbys
            if logging:
                print("Nearbys:")
                print([checkin["name"] for checkin in nearbys])

            possible_checkins, weights, existed_place_ids = get_nearby_parents(nearbys, possible_checkins, weights, existed_place_ids)
            if logging:
                print("Expanded nearbys:")
                print([(checkin["name"], checkin["parent"], weight) for checkin, weight in zip(possible_checkins, weights)])

            parent_weights = [[i] for i in range(len(weights))]
            for i, checkin in enumerate(possible_checkins):
                if checkin["parent_id"] in existed_place_ids:
                    parent_weights[existed_place_ids.index(checkin["parent_id"])].append(i)

            checkin_labels = []
            all_embeddings = []
            embedding_index = []
            filtered_checkins = []

            for i, checkin in enumerate(possible_checkins):
                name = to_english(re.sub("[\(\[].*?[\)\]]", "", checkin["name"]))
                all_label = "I am in a " + ", ".join(checkin["categories"]) + " called " + name
                checkin_labels.append(all_label)
                filtered_checkins.append(checkin)
                if "IMG" in MODES:
                    photos_res = get_photos(checkin["place_id"])
                    checkin["indoor_photos"] = [photo["prefix"] + "original" + photo["suffix"] for photo in photos_res if 'classifications' in photo and 'indoor' in photo["classifications"]]
                    checkin["outdoor_photos"] = [photo["prefix"] + "original" + photo["suffix"] for photo in photos_res if 'classifications' in photo and 'outdoor' in photo["classifications"]]

                    checkin_embeddings, _, _ = get_embeddings(checkin)
                    if checkin_embeddings is not None:
                        embedding_index.append(i)
                        all_embeddings.append(np.mean(checkin_embeddings, axis=0))

            # Text similarity
            text_tokens = clip.tokenize(checkin_labels, truncate=True).cuda()
            with torch.no_grad():
                text_features = model.encode_text(text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)
            mean_similarities = (image_features @ text_features.T).mean(dim=0)

            # Image similarity
            full_image_similarities = torch.clone(mean_similarities)
            if "IMG" in MODES:
                if all_embeddings:
                    checkin_embeddings = torch.tensor(np.stack(all_embeddings)).cuda().float()
                    image_similarities = (image_features @ checkin_embeddings.T).mean(dim=0)
                    for i, similarity in zip(embedding_index, image_similarities):
                        full_image_similarities[i] = (similarity + full_image_similarities[i]) / 2

            weighted_similarities = torch.tensor(weights) * full_image_similarities.cpu()
            mean_probs, mean_labels = (100 * weighted_similarities).softmax(dim=-1).topk(min(len(checkin_labels), 5))

            if logging:
                print("All checkins:")
                print([(checkin["name"], checkin["parent"]) for checkin in possible_checkins])
                print("Photos available:")
                print([possible_checkins[i]["name"] for i in embedding_index])
                print("Weighted similarities")
                print([(weights[label], checkin_labels[label], prob.numpy()) for label, prob in zip(mean_labels, mean_probs)])


            if "REL" not in MODES or mean_probs[0].numpy() > 0.75:
                return filtered_checkins[mean_labels[0]], (mean_probs[0]/weights[mean_labels[0]]).numpy(), mean_labels[0].numpy() < num_nearby_checkins
            else: # Considering parents
                parent_similarities = []
                for ids in parent_weights:
                    sim = 0
                    for i in ids:
                        sim += weighted_similarities[i]
                    parent_similarities.append(sim)
                parent_similarities = torch.tensor(parent_similarities)
                mean_probs, mean_labels = parent_similarities.topk(min(len(checkin_labels), 5))

                if logging:
                    print("Considering parents weights")
                    print([(weights[label], checkin_labels[label], prob.numpy()) for label, prob in zip(mean_labels, mean_probs)])
                return filtered_checkins[mean_labels[0]], (mean_probs[0]/weights[mean_labels[0]]).numpy(), mean_labels[0].numpy() < num_nearby_checkins
    return NULL_CHECKIN, 0, False

# %%
all_checkins = {}
@cache(file_name="cached/checkins")
def save_checkin(checkin):
    global all_checkins
    if checkin["place_id"] not in all_checkins:
        all_checkins[checkin["place_id"]] = checkin
    
# %%
def get_checkin(row, logging=False):
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
#         movement, movement_prob = movement_mode(list(moves.keys()), image_features)
#         movement = moves[movement]
#         if logging:
#             print("Movement:", movement, movement_prob)
#         if movement_prob > 0.7 and movement in ["Inside", "Airport"]:
#             stop = True
#         elif movement_prob > 0.7:
#             stop = False
#         elif max_radius < 100 and movement in ["Inside", "Airport"]: # Low probability
#             stop = True

#         if logging:
#             print("Stop:", stop)

#         stops.loc[row, "stop"] = stop
        if stop:
            if not np.isnan(lat):
                if distance(lat, lon, 53.38998, -6.1457602) < DISTANCE_THRESHOLD:
                    stops.loc[row, "checkin"] = "HOME"
                elif distance(lat, lon, 53.386859863999995, -6.147444621999999) < DISTANCE_THRESHOLD:
                    stops.loc[row, "checkin"] = "Charm Hand & Foot Spa"
                # elif movement_prob > 0.7 and movement in ["Private Home"]:
                #     stops.loc[row, "checkin"] = "Private Home"
                else:
                    checkin, prob, in_checkin = find_named_checkins_nearby(images, image_features, stop, lat, lon, max_radius, logging=logging)
                    save_checkin(checkin)
                    stops.loc[row, "checkin"] = checkin["name"]
                    stops.loc[row, "checkin_id"] = checkin["place_id"]
                    stops.loc[row, "original_name"] = checkin["name"]
                    stops.loc[row, "in_checkin"] = in_checkin
                    stops.loc[row, "categories"] = ", ".join(checkin["categories"])
                    stops.loc[row, "prob"] = prob
                    stops.loc[row, "parent"] = checkin["parent"]
                    stops.loc[row, "parent_id"] = checkin["parent_id"]
            else:
                stops.loc[row, "checkin"] = "Unknown Place"
        # else:
        #     stops.loc[row, "movement"] = movement
        #     stops.loc[row, "movement_prob"] = movement_prob
        if "20200109_133958_000.jpg" in images:
            print(stops.loc[row, "checkin"])

# %% [markdown]
# ## Start

# %%
MODES = ["REL"]
# stops = stops.loc[stops['first'].str.startswith('20190126', na=False)]
# stops = stops.reset_index()
num = len(stops)
print("Total stops:", len(stops))
num_start = 0
num_end = len(stops)
print("Processing from:", num_start, "to", num_end)
for i in tqdm(range(num_start, num_end), desc="Get Checkins"):
    get_checkin(i)

# %%
import importlib
import agg_stops
importlib.reload(agg_stops)

stops = agg_stops.agg_stop(stops)

# %% [markdown]
# # Reclassify Unknown Place

# %%
# stops = pd.read_csv('files/agg_stops.csv', sep=',', decimal='.')
def calculate_distance(checkin, all_lat, all_lon, lat, lon):
    if checkin == "Unknown Place":
        dists = [distance(lt, ln, lat, lon) for (lt, ln) in zip(all_lat, all_lon)]
        dists = [d for d in dists if d]
        if dists:
            return max(dists)
    return 50
tqdm.pandas(desc="Getting radius of stops")
stops["max_radius"] = stops.progress_apply(lambda x: calculate_distance(x['checkin'], x['all_lat'],
                                                               x['all_lon'],
                                                               x['lat'],
                                                               x['lon']), axis=1)
# %%
print("Numbers of Unknown Places", sum(stops['checkin'] == 'Unknown Place'))
for i, checkin in tqdm(enumerate(stops['checkin']), total=len(stops), desc="Processing Unknown Places"):
    if checkin == "Unknown Place":
        get_checkin(i)

# %%
stops = agg_stops.agg_stop(stops)
tqdm.pandas(desc="Getting country for stops")
stops['country'] = stops.progress_apply(lambda x: get_countries(round(x['lat'], 3), round(x['lon'], 3)), axis=1)
# %%
json.dump(all_checkins, open("files/all_checkins.json", "w"), default=str)
stops.to_csv("files/semantic_stops.csv")