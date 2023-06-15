import os
import nibabel as nib
import numpy as np
from PIL import Image
import gzip
from tqdm import tqdm
import json
from datetime import date
import mahotas
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Build annotation', add_help=False)
    parser.add_argument('-s', action='store_true',
                        help="Use shape segmentation")
    parser.add_argument('-o', type=str, default="data",
                        help="directory to outputs")
    return parser

JSON_FILE = 'dataset.json'
TRAINING_DIR = 'train'
VAL_DIR = 'val'
PROB_NO_OBJECT = 0.05


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


def shape_segmentation(gray):
    f = mahotas.gaussian_filter(gray, 4)
    f = (f > f.mean())
    labelled, n_nucleus = mahotas.label(f)
    return labelled


def create_annotation_with_segmentation(label, ni, na, f_name):
    # get image size
    height, width = label.shape[0], label.shape[1]

    # image  annotation dictionary
    image_dict = {
        "id": ni,
        "width": width,
        "height": height,
        "file_name": f_name,
        "date_captured": date.today().strftime("%B %d, %Y")}

    # get unique labels
    unique_labels = np.unique(label).astype(int)

    # iterate all labels
    # checks if there is no object in the image
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        if np.random.rand() < PROB_NO_OBJECT:
            return image_dict, [], True
        else:
            return image_dict, [], False
    annotations = []
    for category in unique_labels[1:]:
        # create annotation for each label

        mask_category = label.copy()
        mask_category[mask_category != category] = 0
        objects = shape_segmentation(mask_category)
        unique_objects = np.unique(objects).astype(int)
        for object in unique_objects[1:]:
            if len(unique_objects) > 2:
                stop = 1
            x, y = np.where(objects == object)
            segmentation = np.stack((x, y), axis=1).astype(np.uint8).flatten().tolist()
            bbox = np.array([np.min(y), np.min(x), np.max(y) - np.min(y), np.max(x) - np.min(x)]).astype(int).tolist()
            area = np.sum(objects == object)
            annotations.append({
                "id": na + len(annotations),
                "image_id": image_dict["id"],
                "category_id": int(category)-1,
                "segmentation": [],
                "area": float(area),
                "bbox": bbox,
                "iscrowd": 0})
    return image_dict, annotations, True


def create_annotation_without_segmentation(label, ni, na, f_name):
    # get image size
    height, width = label.shape[0], label.shape[1]

    # image  annotation dictionary
    image_dict = {
        "id": ni,
        "width": width,
        "height": height,
        "file_name": f_name,
        "date_captured": date.today().strftime("%B %d, %Y")}

    # get unique labels
    unique_labels = np.unique(label).astype(int)

    # iterate all labels
    # checks if there is no object in the image
    if len(unique_labels) == 1 and unique_labels[0] == 0:
        if np.random.rand() < PROB_NO_OBJECT:
            return image_dict, [], True
        else:
            return image_dict, [], False
    annotations = []
    for category in unique_labels[1:]:
        x, y = np.where(label == category)
        bbox = np.array([np.min(y), np.min(x), np.max(y) - np.min(y), np.max(x) - np.min(x)]).astype(int).tolist()
        area = np.sum(label == category)
        annotations.append({
            "id": na + len(annotations),
            "image_id": image_dict["id"],
            "category_id": int(category)-1,
            "segmentation": [],
            "area": float(area),
            "bbox": bbox,
            "iscrowd": 0})
    return image_dict, annotations, True


def build_ds(data, mod, fn, s=0, e=-1):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    # create annotation file and directory
    annotation_dict = {
        "info": {
            "description": "liver dataset",
            "year": 2023,
            "date_created": date.today().strftime("%B %d, %Y")},
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "liver"},
            {"id": 1, "name": "tumor"}]
    }

    # iterate all images in training data

    for ds in tqdm(data["training"][s:e], desc=f"processing {mod} data"):
        img_path, label_path = ds["image"], ds["label"]
        target_dir = os.path.join(args.o, TRAINING_DIR if mod == "train" else VAL_DIR)
        og_f_name = f"{img_path.split('/')[-1].split('.')[0]}"
        # load image and label
        images = gz2nii(img_path)
        labels = gz2nii(label_path)

        # convert to 8-bit unsigned int
        image = np.squeeze(images)  # Remove any unnecessary dimensions
        image = np.clip(image, 0, 255).astype(np.uint8)  # Convert to 8-bit unsigned int

        # iterate all slices of projected image
        for i in range(image.shape[-1]):
            # slice image and label
            img, lbl = image[:, :, i], labels[:, :, i]
            # create annotation
            f_name = og_f_name + f"_{i:04d}.jpg"
            image_d, annotation_d, flag = fn(lbl,
                                             len(annotation_dict["images"]),
                                             len(annotation_dict["annotations"]),
                                             f_name)
            if flag:
                annotation_dict["images"].append(image_d)
                if len(annotation_d) > 0:
                    annotation_dict["annotations"].extend(annotation_d)
                img = Image.fromarray(img)
                img.save(os.path.join(target_dir, f_name), format="JPEG")

    # Writing to json file
    json_object = json.dumps(annotation_dict)
    with open(os.path.join(args.o, "annotations", f"instances_{mod}.json"), "w") as outfile:
        outfile.write(json_object)
    os.rmdir('tmp')

def main(args):
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    if not os.path.exists(os.path.join(args.o, TRAINING_DIR)):
        os.makedirs(os.path.join(args.o, TRAINING_DIR))

    if not os.path.exists(os.path.join(args.o, VAL_DIR)):
        os.makedirs(os.path.join(args.o, VAL_DIR))

    if not os.path.exists(os.path.join(args.o, "annotations")):
        os.makedirs(os.path.join(args.o, "annotations"))

        # read dataset json file
    f = open('dataset.json')
    data = json.load(f)
    f.close()
    fn = create_annotation_with_segmentation if args.s else create_annotation_without_segmentation
    build_ds(data, 'train', fn, 0, 100)
    build_ds(data, 'val', fn, 101, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)


