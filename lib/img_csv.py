import os
import pandas as pd

BASE_DIR = 'images/'

images = [i for i in BASE_DIR]

df = pd.DataFrame()
df['images'] = [BASE_DIR+str(x) for x in images]

pd.to_csv('image_path.csv', header=None)
