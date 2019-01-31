import json
import os
from os.path import expanduser

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state, shuffle

class_name = 'fire hydrant'
size = 64
lw = 1

output_dir = expanduser('~/data/quickdraw/bitmaps')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def draw_cv2(raw_strokes, size=256, lw=1, time_color=True):
    img = np.zeros((256, 256), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != 256:
        return cv2.resize(img, (size, size))
    else:
        return img


random_state = check_random_state(0)

csv_file = expanduser('~/data/quickdraw/fire hydrant.csv')
df = pd.read_csv(csv_file)
df = shuffle(df, random_state=random_state)

df['drawing'] = df['drawing'].apply(json.loads)

samples = {'train': len(df) - 3000, 'test': 3000}
offset = 0

for fold, n_samples in samples.items():
    print(f'====================> {fold}')
    x = np.empty((n_samples, size * size), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.long)

    for i in range(n_samples):
        if i % 100 == 0:
            print(f'Image {i}')
        x[i, :] = draw_cv2(df['drawing'].iloc[i + offset], time_color=False,
                           size=size).ravel()
        x[i, :] /= 255
    offset += n_samples
    joblib.dump((x, y),
                expanduser(f'~/data/quickdraw/bitmaps/'
                           f'{class_name}_{size}_{fold}.pkl'))
