python train.py --exp_name '0222_2s_L' --polyp_dir "data/TrainDataset/" --sam "sam2_small"
python infer.py --exp_name '0222_2s_L' --dataset_name 'CVC-300'  --test_seg_dir "data/TestDataset/CVC-300/"  --sam "sam2_small" 
python infer.py --exp_name '0222_2s_L' --dataset_name 'CVC-ClinicDB'  --test_seg_dir "data/TestDataset/CVC-ClinicDB/"    --sam "sam2_small"
python infer.py --exp_name '0222_2s_L' --dataset_name 'CVC-ColonDB'  --test_seg_dir "data/TestDataset/CVC-ColonDB/"    --sam "sam2_small"
python infer.py --exp_name '0222_2s_L' --dataset_name 'ETIS-LaribPolypDB'  --test_seg_dir "data/TestDataset/ETIS-LaribPolypDB/"    --sam "sam2_small"
python infer.py --exp_name '0222_2s_L' --dataset_name 'Kvasir'  --test_seg_dir "data/TestDataset/Kvasir/"    --sam "sam2_small"
python m_eval.py --exp_name '0222_2s_L'