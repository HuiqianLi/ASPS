# ASPS: Augmented Segment Anything Model for Polyp Segmentation

### Installation

Install the dependencies of [SAM](https://github.com/facebookresearch/segment-anything).

Install mmcv for CNN encoder. 

### Dataset

We conduct extensive experiments on five polyp segmentation datasets
following [PraNet](https://github.com/DengPingFan/PraNet). 

### Training & Infering

We used `train.py` to train and `infer.py` to evaluate our framework. 

The `--exp_name` is the name of the experiment, and `--polyp_dir` is the path to the training dataset.

The `--dataset_name` is the name of the dataset, and `--test_seg_dir` is the path to the testing dataset.

You can directly run the `train.sh` to train and evaluate our framework. 

```bash
python train.py --exp_name '0308_E_L' --polyp_dir "polyp_seg/TrainDataset/"
python infer.py --exp_name '0308_E_L' --dataset_name 'CVC-300'  --test_seg_dir "polyp_seg/TestDataset/CVC-300/"   
python infer.py --exp_name '0308_E_L' --dataset_name 'CVC-ClinicDB'  --test_seg_dir "polyp_seg/TestDataset/CVC-ClinicDB/"   
python infer.py --exp_name '0308_E_L' --dataset_name 'CVC-ColonDB'  --test_seg_dir "polyp_seg/TestDataset/CVC-ColonDB/"   
python infer.py --exp_name '0308_E_L' --dataset_name 'ETIS-LaribPolypDB'  --test_seg_dir "polyp_seg/TestDataset/ETIS-LaribPolypDB/"   
python infer.py --exp_name '0308_E_L' --dataset_name 'Kvasir'  --test_seg_dir "polyp_seg/TestDataset/Kvasir/"   
```

### Vis

To inference single image or visualize the results, run `vis.py`.