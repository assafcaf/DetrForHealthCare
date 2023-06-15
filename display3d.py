import os
import nibabel as nib
import numpy as np
from PIL import Image
import gzip
import cv2
import json
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('-f', type=str, default="liver_1",
                        help="Name of the file to render")
    parser.add_argument('-o', type=str, default="",
                        help="path to output")
    return parser

def gz2nii(file_path):
    output_path = file_path.replace('.nii.gz', '.nii')
    path_broken = output_path.split('/')
    path_broken[1] = 'tmp'
    output_path = '/'.join(path_broken)
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    data = nib.load(output_path).get_fdata()
    os.remove(output_path)
    return data


def main(args):
    # Opening JSON file
    f = open('dataset.json')
    output_dir = args.o
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    for ds in data["training"]:
        img_path, label_path = ds["image"], ds["label"]
        if args.f in img_path:
            images = gz2nii(img_path)
            label = gz2nii(label_path)
            images = np.squeeze(images)  # Remove any unnecessary dimensions
            images = np.clip(images, 0, 255).astype(np.uint8)  # Convert to 8-bit unsigned int
            imgs = []

            f_name = os.path.join(output_dir, f"{img_path.split('/')[-1].split('.')[0]}")
            for i in range(images.shape[-1]):
                img = images[:, :, i]
                lbl = label[:, :, i]
                img = Image.fromarray(img)
                rgbimg = Image.new("RGB", img.size)
                rgbimg.paste(img)
                rgb_image = np.array(rgbimg)
                rgb_image[lbl == 1.] = [0, 255, 0]
                rgb_image[lbl == 2.] = [255, 0, 0]
                imgs.append(Image.fromarray(rgb_image))
            imgs[0].save(f"{f_name}.gif", save_all=True, append_images=imgs[1:], fps=140, duration=0.1)
    os.rmdir("tmp")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)



