# %%
from tzwhere import tzwhere
import bamboolib as bam
import pandas as pd
from gps_utils import *
from map_apis import *
import agg_stops as agg_stops
import json
from unidecode import unidecode
from tqdm.auto import tqdm
import time


# from alphabet_detector import AlphabetDetector
# ad = AlphabetDetector()
# ad.only_alphabet_chars(u"frappé", "LATIN") #True
# ad.is_latin(u"howdy") #True
# from easynmt import EasyNMT
# model = EasyNMT('opus-mt', max_loaded_models=1)
num_requests = 0

def to_english(text, stop, country, debug=False):
    global num_requests
    if isinstance(text, str):
        if stop:
            if text in text_exceptions:
                return text_exceptions[text]
            try:
                if country in ["Ireland"]:  # English speaking countries
                    return unidecode(text)  # No translation
                lang = detect(text)
                if lang != "en":  # TODO!
                    text = translator.translate(text).text
                    num_requests += 1
                    time.sleep(1)
                return unidecode(text)
            except Exception as e:
                if debug:
                    print(f"Error translating \"{text}\"")
                    raise(e)
                else:
                    pass
        return unidecode(text)
    return None
# %%
stops = pd.read_csv("files/semantic_stops.csv")

# %%
tqdm.pandas(desc="Getting country for stops")
stops['country'] = stops.progress_apply(lambda x: get_countries(
    round(x['lat'], 3), round(x['lon'], 3)), axis=1)

# %%
# tqdm.pandas(desc="Translating checkins")
# stops['checkin'] = stops.progress_apply(lambda x: to_english(x['original_name'], x['stop'], x['country']), axis=1)
translations = pd.read_csv("translations.csv", header=None)
translations = {a: b for a, b in zip(translations[0], translations[1])}
stops['checkin'] = stops.progress_apply(
    lambda x: translations[x['original_name']] if x['original_name'] in translations else x['checkin'], axis=1)

# %%
# %%
both = agg_stops.assign_to_images(stops)
tqdm.pandas(desc="Getting cities")
both['city'] = both.progress_apply(lambda x: get_cities(
    round(x['new_lat'], 3), round(x['new_lng'], 3)), axis=1)
tqdm.pandas(desc="Getting countries")
both['country'] = both.progress_apply(lambda x: get_countries(
    round(x['new_lat'], 3), round(x['new_lng'], 3)), axis=1)

tz = tzwhere.tzwhere(forceTZ=True)
tqdm.pandas(desc="Getting timezone")
both["new_timezone"] = both.progress_apply(lambda x: tz.tzNameAt(round(x['new_lat'], 4),
                                                                 round(
                                                                     x['new_lng'], 4),
                                                                 forceTZ=True), axis=1)

# %%
both.to_csv('files/final_metadata.csv')

# %%
# %%
# FOR MYSCEAL
map_visualisation = []
for index, row in stops.iterrows():
    if row["stop"]:
        if isinstance(row["checkin"], str):
            map_visualisation.append(
                (unidecode(row["checkin"]), (row["lat"], row["lon"])))
json.dump(map_visualisation, open(f"files/map_visualisation.json", 'w'))
with open(f"files/commonplace.js", 'w') as f:
    f.write("var commonPlace=" + json.dumps(map_visualisation) +
            ";\n\nexport default commonPlace;")
