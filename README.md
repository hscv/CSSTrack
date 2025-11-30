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

## 3. Hyperspectral Video Dataset
+ The HOT2020 is from "https://www.hsitracking.com/".
+ The IMEC25 dataset is from paper "Histograms of oriented mosaic gradients for snapshot spectral image description".
+ The data should look like:
  ```python
    (1). The format of training dataset:
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
    (2). The format of testing dataset:
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

## 4. Train & Test in HOT2020
(a) cd CSSTrack-HOT2020/

(b) Train: Download pretrained model and put in the folder "pretrained_models", which is available in  
    - https://pan.baidu.com/s/1qRuCKQ2hhE5-MhrkeLiEQA  
    - Access code: 2025
    
    I. Change the path of training data in lib/train/admin/local.py (Line 25: self.hot2020_dir='/data/xx/HOT2020/train')
    II. Run: python tracking/train.py --script csstrack --config CSSTrack-ep30-s256 --save_dir ./output --mode single --nproc_per_node 1

(c) Test: Download testing model of HOT2020 in  
    - https://pan.baidu.com/s/1WJLo72hwzr6y_BtjFFp-Dg
    - Access code: 2025
    
    I. Change the path of training data in lib/train/admin/local.py (Line 20: settings.hot2020_path = '/data/xx/HOT2020/test')
    II. Run: python tracking/test_epoch.py --checkpoint_path ../CSSTrack_ep0030_final.pth.tar

## 5. Train & Testin IMEC25
(a) cd CSSTrack-IMEC25/

(b) Train: Download pretrained model and put in the folder "pretrained_models", which is available in  
    - https://pan.baidu.com/s/1qRuCKQ2hhE5-MhrkeLiEQA
    - Access code: 2025
    
    I. Change the path of training data in lib/train/admin/local.py (Line 25: self.imec25_dir='/data/xxx/HOT/IMEC25/train')
    II. Run: python tracking/train.py --script csstrack --config CSSTrack-ep30-s256 --save_dir ./output --mode single --nproc_per_node 1

(c) Test: Download testing model of IMEC25 in  
    - https://pan.baidu.com/s/1WJLo72hwzr6y_BtjFFp-Dg
    - Access code: 2025
    
    I. Change the path of training data in lib/train/admin/local.py (Line 20: settings.imec25_path = '/data/xxx/HOT/IMEC25/test')
    II. Run: python tracking/test_epoch.py --checkpoint_path ../CSSTrack_ep0030_final.pth.tar


## 6. Concat
* lizhuanfeng@hytc.edu.cn;
* If you have any questions, just contact me.


## Acknowledgments
* Thanks for the [AQATrack](https://github.com/GXNU-ZhongLab/AQATrack) and [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas.
