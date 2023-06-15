


# Hw 2 AI for HealthCare Detr 


This repository contains code for training 'Detr' model to detect liver and cancer from CT images, including the preprocess and and the building of the dataset based on [COCO](https://cocodataset.org/#home) format taken from [medicaldecathlon](https://drive.google.com/file/d/1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu/view?usp=drive_link).
The code for the Detr model is based on the following repo [Detr Repo](https://github.com/aivclab/detr/tree/master).

---
**Instolation**: 


1. cerate new conda environment using `conda create -n <env_name> pytohn=3.9`
2. activate env using `conda activate <env_name>`
3. run `pip install -f requirements.txt`
4. this repo is built under the assumption that the data files will be extracted to repo directory
* This installation does not include a torch version that supports Cuda. If there is cuda enabled card on the machine pls install the proper version of Cuda from here [Torch](https://pytorch.org/get-started/locally/) and run the cude with `--cuda` flag to utilize the graphic card
---
**Building COCO dataset**: 

 1. make sure that the data from [medicaldecathlon](https://drive.google.com/file/d/1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu/view?usp=drive_link) is extracted in repo directory
 2. run `build_annotations.py` with the `-o <name_dir>` flag for running without Segmentation of Multiple Targets or `-s` for running with Segmentation of Multiple Targets

make sure that a new directory is created under the `name_dir` with the following structure

    dir_name/
      annotations/  # annotation json files
      train/        # train images
      val/          # val images
---
**Display gif with annotations**: 

Run `display3d.py -r <liver_X>`
For example `display3d.py -f <liver_18>`

![Example](https://github.com/assafcaf/DetrForHealthCare/blob/main/images/liver_53.gif)
---

**Training**

For default parameter settings run `main.py --coco_path <data_name_dir>`  such that `<data_name_dir>` is the path to the outputs from `build_annotations.py` file

For more advanced parameter settings try `main.py -h`


---
**Predictions**
run `make_predict.py` with `-d <path_to_data> -m <path_to_model>`
for example `python make_predict.py -d data -m outputs/model_X`

![prediction](https://github.com/assafcaf/DetrForHealthCare/blob/main/images/prediction.png)


---
**Model results without shape segmentation**
| Metric | results 
|--|--|
| Average Precision  (AP) @[ IoU=0.50:0.95 / area=   all / maxDets=100 ]|0.337|
| Average Precision  (AP) @[ IoU=0.50      / area=   all / maxDets=100 ] |0.513|
| Average Precision  (AP) @[ IoU=0.75      / area=   all / maxDets=100 ]| 0.384|
| Average Precision  (AP) @[ IoU=0.50:0.95 /area= small / maxDets=100 ]|0.103|
| Average Precision  (AP) @[ IoU=0.50:0.95 / area=medium /maxDets=100 ]|0.323|
| Average Precision  (AP) @[ IoU=0.50:0.95 / area= large / maxDets=100 ]|0.366|
| Average Recall     (AR) @[ IoU=0.50:0.95 / area=   all / maxDets=  1 ]|0.402|
| Average Recall     (AR) @[ IoU=0.50:0.95 / area=   all / maxDets= 10 ]|0.402|
| Average Recall     (AR) @[ IoU=0.50:0.95 / area=   all / maxDets=100 ]|0.402|
|Average Recall     (AR) @[ IoU=0.50:0.95 / area= small / maxDets=100 ]|0.119|
| Average Recall     (AR) @[ IoU=0.50:0.95 / area=medium / maxDets=100 ]|0.376|
|Average Recall     (AR) @[ IoU=0.50:0.95 / area= large / maxDets=100 ]|0.477|
