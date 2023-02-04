import requests
from gps_utils import cache
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

locator = Nominatim(user_agent="myGeocoder")
rgeocode = RateLimiter(locator.reverse, min_delay_seconds=0.001)
FourSpace_API = 'fsq3rPKMEwa25FipLv2i+FoGKeSbDM1Hg2BBw8I9+ektlKM='


@cache(file_name='cached/fourspace_places')
def get_categories(place_id):
    url = f"https://api.foursquare.com/v3/places/{place_id}?fields=description%2Crelated_places%2Ccategories"
    headers = {
        "Accept": "application/json",
        "Authorization": FourSpace_API
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(response.text)
    return {"categories": []}


@cache(file_name='cached/fourspace_photos')
def get_photos(place_id):
    url = f"https://api.foursquare.com/v3/places/{place_id}/photos?classifications=indoor%2Coutdoor"
    headers = {
        "Accept": "application/json",
        "Authorization": FourSpace_API
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(response.text)
        print(place_id)
    return []


@cache(file_name="cached/fourspace_nearby_distance")
def get_nearby_places(lat, lon):
    url = f"https://api.foursquare.com/v3/places/nearby?fields=distance%2Crelated_places%2Cname%2Cfsq_id%2Ccategories%2Cdescription&ll={lat}%2C{lon}"

    headers = {
        "Accept": "application/json",
        "Authorization": FourSpace_API
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(response.text)
        print(lat, lon)
    return []

def parse_checkin(res):
    checkin = {"name": res["name"],
               "place_id": res["fsq_id"]}
    checkin["categories"] = [cat['name'] for cat in res["categories"]]
#     print(categories_res)
    if "description" in res:
        checkin["description"] = res["description"]
    else:
        checkin["description"] = ""
    if "related_places" in res and "parent" in res["related_places"]:
        checkin["parent"] = res["related_places"]["parent"]["name"]
        checkin["parent_id"] = res["related_places"]["parent"]["fsq_id"]
    else:
        checkin["parent"] = ""
        checkin["parent_id"] = ""
    return checkin


@cache(file_name="cached/cities")
def get_cities(lat, lon):
    # coordinates = "10.76833026, 106.67583063"
    all_infos = []
    coordinates = f"{lat} {lon}"
    try:
        location = rgeocode(coordinates, language='en')
    except ValueError as e:
        print(lat, lon)
    if location:
        address = location.raw['address']
        for label in ["county", "city", "state", "country", "region"]:
            if label in address and address[label] not in all_infos:
                all_infos.append(address[label].replace(
                    label.capitalize(), "").strip())
    return ", ".join(all_infos)

@cache(file_name="cached/countries")
def get_countries(lat, lon):
    coordinates = f"{lat} {lon}"
    try:
        location = rgeocode(coordinates, language='en')
    except ValueError as e:
        print(lat, lon)
    if location:
        address = location.raw['address']
        if "country" in address:
            return address["country"].replace("Country", "").strip()
    return None