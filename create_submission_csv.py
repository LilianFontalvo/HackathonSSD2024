import argparse
import os
import cv2
import numpy as np
import pandas as pd

def rle_encode(img):
    '''Run length encoding
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[:-1:2]
    
    return ' '.join(str(x) for x in runs)

def create_csv(masks_dir, output_dir):
    l_files = os.listdir(masks_dir)

    dict_df = {'img_key': [], 'rle_mask': []}
    for file_key in l_files:
        img_array = cv2.imread(os.path.join(masks_dir, file_key), cv2.IMREAD_GRAYSCALE)
        img_array //= 255
        rle_mask = rle_encode(img_array)
        dict_df['rle_mask'].append(rle_mask)
        dict_df['img_key'].append(file_key)
    df = pd.DataFrame(dict_df)
    df.to_csv(output_dir, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to create the submission csv for Kaggle comp.",
        add_help=True,
    )

    parser.add_argument("masks_dir", default=None, help="The path to the folder containing the test set masks.")
    parser.add_argument('-c', "--csv", default='./submission.csv', help="The output csv path.")
    args = parser.parse_args()

    create_csv(args.masks_dir, args.csv)