
import bisect
from datetime import datetime
import geopy.distance
from collections import Counter
import shelve
import numpy as np
import json
from dateutil import tz


def cache(_func=None, *, file_name=None, separator='_'):
    """
    if file_name is None, just cache it using memory, else save result to file
    """
    if file_name:
        d = shelve.open(file_name)
    else:
        d = {}

    def decorator(func):
        def new_func(*args, **kwargs):
            param = separator.join(
                [str(arg) for arg in args] + [str(v) for v in kwargs.values()])
            if param not in d:
                d[param] = func(*args, **kwargs)
#             else:
#                 print("Use cached value")
            return d[param]
        return new_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)

from googletrans import Translator, constants
from langdetect import detect
# %%
# Translating
translator = Translator()
text_exceptions={"Adapt Centre @ Dcu": "Adapt Centre @ Dcu",
                "777": "777",
                "Nip@tuck": "Nip@tuck"}
def to_english(text, debug=False):
    if text:
        if text in text_exceptions:
            return text_exceptions[text]
        try:
            lang = detect(text)
            if lang != "en":
                text = translator.translate(text).text
            return unidecode(text)
        except Exception as e:
            if debug:
                print(f"Error translating \"{text}\"")
                raise(e)
            else:
                pass
    return text

def smooth(series, window_size=3):
    smoothed_series = []
    appended_series = list(series)
    appended_series = [appended_series[0]] * (window_size//2) + appended_series + [appended_series[-1]] * (window_size//2)
    for i in range(len(series)):
        window = [appended_series[i+window_size//2]] + appended_series[i:i+window_size//2] + appended_series[i+window_size//2+1:i+window_size]
        assert len(window) == window_size, "Not equal window size"
        smoothed_series.append(Counter(window).most_common(1)[0][0])

    return smoothed_series


def distance(lt1, ln1, lt2, ln2):
    try:
        return(geopy.distance.distance([lt1, ln1], [lt2, ln2]).m)
    except Exception as e:
        return None


def to_date(minute_id):
    return datetime.strptime(minute_id+"00", "%Y%m%d_%H%M%S")


def list_all(series):
    return list(series.dropna())


def unique_all(series):
    return set(list(series))


def list_filter(series):
    all_names = list(series.dropna())
    return [", ".join(name.split(',')[:2]) for name in all_names]


count_none = 0
def most_common(series):
    global count_none
    all_names = list(series.dropna())
    all_names = [",".join(name.split(',')[:2]) for name in all_names]
    if all_names:
        return Counter(all_names).most_common(1)[0][0]
    count_none += 1
    return np.nan
    # return f"Unknown Place {count_none}"


def unpack_str(series):
    final = []
    for array in series:
        if isinstance(array, str):
            if array:
                final.extend(json.loads(array.replace("'", '"')))
    return final


def count_unpack_str(series):
    final = []
    for array in series:
        if isinstance(array, str):
            if array:
                final.extend(json.loads(array.replace("'", '"')))
    return len(final)

def unpack(series):
    return [item for array in series for item in array]


def mean_unpack(series):
    return np.mean([item for array in series for item in array])


def to_date(minute_id):
    return datetime.strptime(minute_id+"00", "%Y%m%d_%H%M%S")

def image_to_date(image_id):
    return datetime.strptime(image_id, "%Y%m%d_%H%M%S_000.jpg")

def image_time_duration(series):
    series = list(series)
    return (image_to_date(series[-1]) - image_to_date(series[0])).seconds/60 + 1


def time_duration(series):
    series = list(series)
#     20190101_1142
    return (to_date(series[-1]) - to_date(series[0])).seconds/60 + 1


def time_duration_unpack(series):
    series = list(series)
#     20190101_1142
    return f"{(to_date(series[-1][-1]) - to_date(series[0][0])).seconds/60}"


def unique_all(series):
    series = list(set([s for s in series]))
    if len(series) == 1:
        return series[0]
    if series:
        return series
    else:
        return np.nan


def find_closest_index(a, x): # to the right only
    i = bisect.bisect_left(a, x)
    if i >= len(a):
        i = len(a) - 1
    return i
