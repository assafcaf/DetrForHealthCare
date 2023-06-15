import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import argparse

labels_map = {0: ['liver', 'GnBu', 'lightgreen'], 1: ['cancer', 'BuPu', 'skyblue']}


def get_args_parser():
    parser = argparse.ArgumentParser('EDA', add_help=False)
    parser.add_argument('-d', type=str, default="",
                        help="path to data directory")
    return parser


def gen_eda_plots(annotations, test=False):
    """
    Plots three graphs per dataset:
    1. Pie chart of the label distribution in the data
    2. Pie chart of the distinct number of images each labels appears in
    3. A box plot of the areas of each label
    4. A histogram of the number of distinct objects, per label, that exist in each image
    3. A heap map of the bounding boxes of each label
    """
    df = pd.DataFrame(annotations)
    dataset = ' - Test Data' if test else ' - Train Data'

    labels = ['Liver Areas', 'Cancer Areas']
    colors = ['lightgreen', 'skyblue']

    labels_distribution = df.category_id.value_counts(dropna=False)
    gen_bar_chart(colors, dataset, labels, labels_distribution, distinct_flag=False)

    distinct_labels_distribution = df.groupby('category_id')['image_id'].nunique()
    gen_bar_chart(colors, dataset, labels, distinct_labels_distribution, distinct_flag=True)


    gen_box_plot(colors, dataset, df, labels)

    gen_object_freq_hist(dataset, df)

    gen_heat_map(dataset, df)


def gen_heat_map(dataset, df):
    grid_size = 512
    heatmap = np.zeros((grid_size, grid_size))
    for cat in range(df.category_id.nunique()):
        dff = df[df.category_id == cat]
        for bbox in dff['bbox']:
            x_min, y_min, h, w = bbox

            # Calculate the indices of the region covered by the bounding box
            x_start = max(0, x_min)
            x_end = min(grid_size, x_min + w)
            y_start = max(0, y_min)
            y_end = min(grid_size, y_min + h)

            # Increment the corresponding elements in the grid
            heatmap[y_start:y_end, x_start:x_end] += 1

        heatmap /= np.max(heatmap)

        plt.imshow(heatmap, cmap=labels_map[cat][1], interpolation='nearest')

        plt.colorbar(label='Frequency')
        plt.title(f'Heat Map of {labels_map[cat][0]} {dataset}'.title())
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(f'Heatmap - {labels_map[cat][0]} {dataset}')
        plt.show()


def gen_object_freq_hist(dataset, df):
    for cat in range(df.category_id.nunique()):
        dff = df[df.category_id == cat]
        gb = dff.groupby('image_id')['id'].count().sort_values(ascending=False)

        plt.hist(gb, bins=10, edgecolor='black', color=labels_map[cat][2])

        plt.xlabel('Values')
        plt.ylabel('Counts')
        plt.title('Histogram')

        plt.grid(axis='y', alpha=0.5)

        plt.gca().set_facecolor('#EEEEEE')

        plt.xticks(range(1, 10))
        plt.yticks(range(0, 6))
        plt.title(f'Num Objects Per Image - {labels_map[cat][0]} {dataset}'.title())

        plt.savefig(f'Num Objects Per Image - {labels_map[cat][0]} {dataset}')
        plt.show()


def gen_box_plot(colors, dataset, df, labels):
    areas = [list(df[df.category_id == 0]['area']), list(df[df.category_id == 1]['area'])]
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size
    boxplot = ax.boxplot(areas, labels=labels, patch_artist=True, notch=True)
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.yaxis.grid(True)
    plt.title(f'Distribution of Liver and Cancer Areas {dataset}')
    plt.xlabel('Categories')
    plt.ylabel('Areas')
    for whisker in boxplot['whiskers']:
        whisker.set(color='gray', linestyle='--', linewidth=1)
    for cap in boxplot['caps']:
        cap.set(color='gray', linewidth=2)
    for flier in boxplot['fliers']:
        flier.set(marker='o', color='red', alpha=0.5)
    plt.savefig(f'Boxplot {dataset}')
    plt.show()


def gen_bar_chart(colors, dataset, labels, labels_distribution, distinct_flag):
    title = f'Labels Distribution {dataset}' if not distinct_flag else f'Labels Distribution In Distinct Images {dataset}'
    plt.pie(labels_distribution, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.title(title.title())
    plt.savefig(title)
    plt.show()


def main(args):
    ann_pth = args.d
    with open(os.path.join(ann_pth, "instances_train.json"), 'r') as f:
        train = json.load(f)
    with open(os.path.join(ann_pth, "instances_val.json"), 'r') as f:
        val = json.load(f)
    train_annotations = train['annotations']
    test_annotations = val['annotations']

    gen_eda_plots(train_annotations)
    gen_eda_plots(test_annotations, test=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('display predictions', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
