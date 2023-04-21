from pathlib import Path
import os
import pandas as pd
# import open_clip
import torch
from PIL import Image
import math
import numpy as np
import clip
from tqdm import tqdm
import json

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(name, pretrained):
    # model, _, preprocess = open_clip.create_model_and_transforms(name, 
    #                                                              pretrained=pretrained,
    #                                                              device=device,
    #                                                              jit=True)
    model, preprocess = clip.load(name, device=device)
    return model, preprocess

def compute_clip_features(photos_batch):
    # Load all the photos from the files
    photos = [Image.open(photo_file) for photo_file in photos_batch]

    # Preprocess all photos
    photos_preprocessed = torch.stack(
        [preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        # Encode the photos batch to compute the feature vectors and normalize them
        photos_features = model.encode_image(photos_preprocessed)
        # photos_features /= photos_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return photos_features.cpu().numpy()

if __name__ == "__main__":
    # OLD PHOTOS
    # print("Available models", open_clip.list_pretrained())
    all_images = pd.read_csv('vaisl_gps.csv')["ImageID"].tolist()
    photofiles2 = []
    photo_keys = []
    photos_path = '/mnt/DATA/duyen/highres/LSC23/'
    model_name = "ViT-L-14-336"
    pretrained = "openai"
    output_path = f'/mnt/DATA/duyen/highres/LSC23/{model_name.replace("/", "-")}_{pretrained}_nonorm'
    batch_size = 48
    
    print("Looking up photos")
    for image in tqdm(all_images):
        if isinstance(image, str):
            path = f"{photos_path}/{image[:6]}/{image[6:8]}/{image}"
            if os.path.exists(path):
                photofiles2.append(path)
                photo_keys.append(f"{image[:6]}/{image[6:8]}/{image}")
    print("Loading model", model_name, pretrained)
    model, preprocess = load_model(model_name, pretrained)

    features_path = Path(output_path)
    os.system(f"mkdir {output_path}")
    batches = math.ceil(len(photofiles2) / batch_size)
    for i in tqdm(range(batches)):
        batch_ids_path = features_path / f"{i:010d}.csv"
        batch_features_path = features_path / f"{i:010d}.npy"

        # Only do the processing if the batch wasn't processed yet
        if not batch_features_path.exists():
            try:
                # Select the photos for the current batch
                batch_files = photofiles2[i*batch_size: (i+1)*batch_size]

                # Compute the features and save to a numpy file
                batch_features = compute_clip_features(batch_files)
                np.save(batch_features_path, batch_features)

                # Save the photo IDs to a CSV file
                photo_ids = photo_keys[i*batch_size: (i+1)*batch_size]
                photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
                photo_ids_data.to_csv(batch_ids_path, index=False)
            except Exception as e:
                # Catch problems with the processing to make the process more robust
                print(f'Problem with batch {i}')
                raise(e)

    features_list = [np.load(features_file)
                     for features_file in sorted(features_path.glob("*.npy"))]

    # Concatenate the features and store in a merged file
    features = np.concatenate(features_list)
    np.save(features_path / "features.npy", features)

    photo_ids = pd.concat([pd.read_csv(ids_file)
                           for ids_file in sorted(features_path.glob("*.csv"))])
    photo_ids.to_csv(features_path / "photo_ids.csv", index=False)
    os.system(f"rm {output_path}/0*")
