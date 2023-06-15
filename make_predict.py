import argparse
import json
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from models import build_model
import torchvision.transforms as T
import matplotlib.patches as patches
import random
from argparse import Namespace
import gdown
import zipfile


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CLASSES = ['liver', 'cancer']


def get_args_parser():
    parser = argparse.ArgumentParser('display predictions', add_help=False)
    parser.add_argument('-r', type=int, default="1",
                        help="amount of rows int the plot")

    parser.add_argument('-c', type=int, default="2",
                        help="amount of columns int the plot")

    parser.add_argument('-m', type=str, default="",
                        help="path to model directory")

    parser.add_argument('-d', type=str, default="",
                        help="path to data directory")

    return parser


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes, ax):
    ax.imshow(pil_img, cmap='gray')
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=3, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=10,
                bbox=dict(facecolor='yellow', alpha=0.5))
    ax.axis('off')


def add_annotation(annotations, ax):
    for ann in annotations:
        xmin, ymin, xmax, ymax = ann['bbox']
        rect = patches.Rectangle((xmin, ymin),  xmax, ymax, linewidth=3, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

        text = f'{CLASSES[ann["category_id"]]}'
        ax.text(xmin+xmax, ymin, text, fontsize=10,
                bbox=dict(facecolor='green', alpha=0.5))


def main(args):
    # pth
    if args.m != "":
        model_pth = args.m
    else:
        model_pth = os.path.join("outputs", args.m, "model_1")
    url = "https://drive.google.com/file/d/1DgGc3VGUPFzkQPtxk6WeC3jMhW1URS4k/view?usp=drive_link"

    if not os.path.isdir(model_pth):
        os.mkdir(model_pth)

    ds_pth = args.d
    mode = "val"
    imgs_pth = os.path.join(ds_pth, mode)
    annotations_pth = os.path.join(ds_pth, "annotations", f"instances_{mode}.json")

    # download model weights
    if args.m == "":
        model_name = "model_1"
        gdown.download(url, os.path.join(model_pth, "model.zip"), quiet=False, fuzzy=True)
        with zipfile.ZipFile(os.path.join(model_pth, "model.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(model_pth))

    # load annotations
    with open(annotations_pth, 'r') as f:
        data = json.load(f)

    # load model
    with open(os.path.join(model_pth, 'args.json'), 'r') as f:
        args1 = Namespace(**json.load(f))
    model, criterion, postprocessors = build_model(args1)


    chk = torch.load(os.path.join(model_pth, "checkpoint.pth"))
    model.load_state_dict(chk["model"])
    model.eval()

    # data transform
    transform = T.Compose([
        T.Resize(512),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    fig, axs = plt.subplots(args.r, args.c, figsize=(16, 12))
    for ax in axs.reshape(-1):
        img_pth = random.choice(os.listdir(imgs_pth))
        im_id = [im["id"] for im in data["images"] if im["file_name"] == img_pth][0]
        annotations = [ann for ann in data["annotations"] if ann["image_id"] == im_id]

        im = Image.open(os.path.join(imgs_pth, img_pth))
        img = transform(im.convert('RGB')).unsqueeze(0)
        # propagate through the model
        outputs = model(img)

        # keep only predictions with 0.7+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > .9

        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
        plot_results(im, probas[keep], bboxes_scaled, ax)
        add_annotation(annotations, ax)
        ax.set_title(img_pth)



    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('display predictions', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
