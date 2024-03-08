python train.py --exp_name '0217' --polyp_dir 'data/TrainDataset' --dataset_name 'TrainDataset'
python infer.py --exp_name '0217' --dataset_name 'TestHardDataset/Unseen'  --test_seg_dir "data/TestHardDataset/Unseen/"
python infer.py --exp_name '0217' --dataset_name 'TestHardDataset/Seen'  --test_seg_dir "data/TestHardDataset/Seen/"
python infer.py --exp_name '0217' --dataset_name 'TestEasyDataset/Unseen'  --test_seg_dir "data/TestEasyDataset/Unseen/"
python infer.py --exp_name '0217' --dataset_name 'TestEasyDataset/Seen'  --test_seg_dir "data/TestEasyDataset/Seen/"