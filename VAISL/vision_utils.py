import requests
import clip
import torch
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get checkins embeddings
model, preprocess = clip.load("ViT-L/14@336px")
model.cuda().eval()
print("")

moves = {"I am sitting on an airplane": "Airplane",
         "I am in a car": "Car",
         "I am in an airport": "Inside",
         "I am cycling": "Cycling",
         "I am walking outside or on the street": "Walking Outside",
         "I am on public transport": "Public Transport",
         "I am inside a building or a house": "Inside"}

insides = {"I am inside a building or a house": "Inside",
           "I am outside": "Outside",
           "I am in a transport": "Transport"}

def validate(images, url=False):
    photos = []
    for image_path in images:
        try:
            if url:
                response = requests.get(image_path)
                photo = Image.open(BytesIO(response.content))
            else:
                photo = Image.open(image_path)
            photos.append(photo)
        except IOError as e:
            continue
    return photos


def compute_clip_features(model, preprocess, photos_batch, url=False):
    # Load all the photos from the files
    photos = validate(photos_batch, url)
    if photos:
        # Preprocess all photos
        photos_preprocessed = torch.stack(
            [preprocess(photo) for photo in photos]).to(device)

        with torch.no_grad():
            # Encode the photos batch to compute the feature vectors and normalize them
            photos_features = model.encode_image(photos_preprocessed)
            photos_features /= photos_features.norm(dim=-1, keepdim=True)

        # Transfer the feature vectors back to the CPU and convert to numpy
        return photos_features.cpu().numpy()
    else:
        return None


# @cache(file_name='cached/embeddings')
def get_embeddings(checkin):
    all_photos = []
    if "indoor_photos" in checkin:
        # indoor_embeddings = compute_clip_features(
        #     model, preprocess, checkin["indoor_photos"], url=True)
        all_photos.extend(checkin["indoor_photos"])
    if "outdoor_photos" in checkin:
        all_photos.extend(checkin["outdoor_photos"])
    # outdoor_embeddings = compute_clip_features(
    #     model, preprocess, checkin["outdoor_photos"], url=True)
    all_embeddings = compute_clip_features(
        model, preprocess, all_photos, url=True)
    return all_embeddings, None, None


photo_features = np.load("/home/tlduyen/LSC22/process/files/embeddings/features.npy")
photo_ids = list(pd.read_csv(
    "/home/tlduyen/LSC22/process/files/embeddings/photo_ids.csv")["photo_id"])
clip_embeddings = {photo_id: photo_feature for photo_id,
                   photo_feature in zip(photo_ids, photo_features)}
photo_features = None
photo_ids = None

# @cache(file_name='cached/stops_embeddings')
def get_stop_embeddings(photos):
    photos = [f"{photo[:6]}/{photo[6:8]}/{photo}" for photo in photos]
    embeddings = [clip_embeddings[photo]
                  for photo in photos if photo in clip_embeddings]
    if embeddings:
        return np.stack(embeddings)
    else:
        return None


def movement_mode(moves, image_features):
    text_tokens = clip.tokenize(moves, truncate=True).cuda()

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    mean_probs, mean_labels = (100 * image_features @
                               text_features.T).mean(dim=0).softmax(dim=-1).cpu().topk(min(len(moves), 5))
    return moves[mean_labels[0]], mean_probs[0].numpy()
