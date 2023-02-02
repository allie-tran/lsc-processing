from tqdm.auto import tqdm
import pandas as pd
from vision_utils import *
from gps_utils import smooth

visual_file = '../../original_data/lsc22_visual_concepts.csv'
visual = pd.read_csv(visual_file, sep=',', decimal='.')

# %%
# Assign movement
visual["movement"] = [None for i in range(len(visual))]
visual["movement_prob"] = [0 for i in range(len(visual))]
for i, row in tqdm(visual.iterrows(), total=len(visual), desc='assign movement'):
    image_features = get_stop_embeddings([row["ImageID"]])
    try:
        image_features = torch.tensor(image_features).cuda().float()
    except RuntimeError as e:
        continue
    movement, prob = movement_mode(list(moves.keys()), image_features)
    visual.loc[i, "movement"] = moves[movement]
    visual.loc[i, "movement_prob"] = prob
visual.to_csv("files/visual_with_movement.csv")

# %%
theta = 0.9
visual["movement"] = smooth(visual["movement"], 3)
visual["inside"] = visual["movement"] == "Inside"
# visual.loc[(visual["inside"] == False) & (visual["movement_prob"] > theta), 'stop'] = False
# visual.loc[(visual["inside"] == False) & (visual["movement_prob"] > theta), 'stop_label'] = ""
visual.to_csv("files/visual_with_movement.csv")
