# CSSTrack  

# Quick Start

## The soure code of the paper "CSSTrack".

## 1. Install the environment

Use the Anaconda
```
conda create -n csstrack python=3.8
conda activate csstrack
bash install.sh
```

## 2. Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## 2. Hyperspectral Video Dataset
+ The HOT2020 is from "https://www.hsitracking.com/".
+ The IMEC25 dataset is from paper "Histograms of oriented mosaic gradients for snapshot spectral image description".
+ The data should look like:
  ```
    1. The format of training dataset:
        rootDir |-
                   videoName1
                       |- HSI
                           |- 0001.png
                           |- 0002.png
                           ...
                           |- XXXX.png
                           |- groundturth_rect.txt
                   videoName2
                       |- HSI
                           |- 0001.png
                           |- 0002.png
                           ...
                           |- XXXX.png
                           |- groundturth_rect.txt
                   ...
                   videoNameN
                       |- HSI
                           |- 0001.png
                           |- 0002.png
                           ...
                           |- XXXX.png
                           |- groundturth_rect.txt
    ```
    ```python
    2. The format of testing dataset:
        rootDir |-
                   test_HSI
                       |- videoName1
                           |- groundturth_rect.txt
                           |- HSI
                                |- 0001.png
                                |- 0002.png
                                |- ...
                                |- XXXX.png
                       |- videoName2
                           |- groundturth_rect.txt
                           |- HSI
                                |- 0001.png
                                |- 0002.png
                                |- ...
                                |- XXXX.png
                       ...
                       |- videoNameM
                           |- groundturth_rect.txt
                           |- HSI
                                |- 0001.png
                                |- 0002.png
                                |- ...
                                |- XXXX.png
    ```

## 4. Train in HOT2020
(a) Download pretrained model and put in the folder "pretrained_models", which is available in  
    - https://pan.baidu.com/s/1qRuCKQ2hhE5-MhrkeLiEQA
    - Access code: 2025    
(b) cd CSSTrack-HOT2020/
(c) Change the path of training data in lib/train/admin/local.py (Line 100: settings.env.hsi_dir='/data/XXX/XX')
(d) Run: python tracking/train.py --script vipt --config deep_all --save_dir ./output

## 5. Train in IMEC25
(a) Download pretrained model and put in the folder "pretrained_models", which is available in  
    - https://pan.baidu.com/s/1qRuCKQ2hhE5-MhrkeLiEQA
    - Access code: 2025    
(b) cd CSSTrack-HOT2020/
(c) Change the path of training data in lib/train/admin/local.py (Line 100: settings.env.hsi_dir='/data/XXX/XX')
(d) Run: python tracking/train.py --script vipt --config deep_all --save_dir ./output


## 6. Test in HOT2020
(a) Download testing model of HOT2020 in  
    - https://pan.baidu.com/s/1WJLo72hwzr6y_BtjFFp-Dg
    - Access code: 2025
    
(b) cd CSSTrack-HOT2020/ & Put the testing model in the folder "final_model".

(c) Run:
```
python test_hsi_mgpus_all.py --dataset_name HOT23TEST --data_path /data/lizf/HOT/Whispers2023/validation/HSI-VIS --model_path final_model_path_HOT2023
```

## 7. Test in IMEC25
(a) Download testing model in  
    - https://pan.baidu.com/s/1WJLo72hwzr6y_BtjFFp-Dg
    - Access code: 2025
    
(b) Put the testing model in the folder "final_model".

(c) Run in HOT2023:
```
VIS domain: python test_hsi_mgpus_all.py --dataset_name HOT23TEST --data_path /data/lizf/HOT/Whispers2023/validation/HSI-VIS --model_path final_model_path_HOT2023
NIR domain: python test_hsi_mgpus_all.py --dataset_name HOT23TEST --data_path /data/lizf/HOT/Whispers2023/validation/HSI-NIR --model_path final_model_path_HOT2023
RedNIR domain: python test_hsi_mgpus_all.py --dataset_name HOT23TEST --data_path /data/lizf/HOT/Whispers2023/validation/HSI-RedNIR --model_path final_model_path_HOT2023
```


## 7. Cite
```
@article{LI2025111389,
title = {Multi-domain universal representation learning for hyperspectral object tracking},
author = {Zhuanfeng Li and Fengchao Xiong and Jianfeng Lu and Jing Wang and Diqi Chen and Jun Zhou and Yuntao Qian},
journal = {Pattern Recognition},
volume = {162},
pages = {111389},
year = {2025},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.111389},
}
```


## 8. Concat
* lizhuanfeng@njust.edu.cn;
* If you have any questions, just contact me.



## Install the environment
Use the Anaconda
```
conda create -n aqatrack python=3.8
conda activate aqatrack
bash install.sh
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training
Download pre-trained [HiViT-Base weights](https://drive.google.com/file/d/1VZQz4buhlepZ5akTcEvrA3a_nxsQZ8eQ/view?usp=share_link) and put it under `$PROJECT_ROOT$/pretrained_models` (see [HiViT](https://github.com/zhangxiaosong18/hivit) for more details).

```
bash train.sh
```


## Test
```
python test_epoch.py
```

## Evaluation 
```
python tracking/analysis_results.py
```


## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX2080Ti GPU.

```
# Profiling AQATrack-ep150-full-256
python tracking/profile_model.py --script aqatrack --config AQATrack-ep150-full-256
# Profiling AQATrack-ep150-full-384
python tracking/profile_model.py --script aqatrack --config AQATrack-ep150-full-384
```


## Acknowledgments
* Thanks for the [EVPTrack](https://github.com/GXNU-ZhongLab/EVPTrack) and [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas.


## Citation
If our work is useful for your research, please consider cite:

```
@inproceedings{xie2024autoregressive,
  title={Autoregressive Queries for Adaptive Tracking with Spatio-Temporal Transformers},
  author={Xie, Jinxia and Zhong, Bineng and Mo, Zhiyi and Zhang, Shengping and Shi, Liangtao and Song, Shuxiang and Ji, Rongrong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19300--19309},
  year={2024}
}
```
