import os
import pandas as pd

BASE_DIR = 'images/'
Base = sorted(os.listdir(BASE_DIR))
images = [i for i in Base]

df = pd.DataFrame()
df['images'] = [BASE_DIR+str(x) for x in images]

pd.to_csv('image_path.csv', header=None)
