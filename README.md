# ASPS: Augmented Segment Anything Model for Polyp Segmentation

### News

2024/6/25: 🎉Our method was accepted by **MICCAI 2024**.

2024/5/21: Add data loader for Skin Lesion Segmentation (ISIC2017).

### Installation

Install the dependencies of [SAM](https://github.com/facebookresearch/segment-anything).

Install mmcv for CNN encoder. 

### Dataset

We conduct extensive experiments on five polyp segmentation datasets
following [PraNet](https://github.com/DengPingFan/PraNet). 

>  For skin lesion segmentation: following [EGE-UNet](https://github.com/JCruan519/EGE-UNet), needing to modify `from dataset.Segmentation_other` to `from dataset.Segmentation_isic` both in `train.py` and `infer.py`.

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