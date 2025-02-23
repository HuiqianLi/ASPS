import time
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

from option import args
from dataset.Segmentation_other import DatasetSegmentationInfer
from model.Network import Network
from segment_anything import sam_model_registry
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from sam2.build_sam import build_sam2

def compute_dice(seg, gt):
    intersection = torch.sum(seg * gt)
    union = torch.sum(seg) + torch.sum(gt)
    return (2. * intersection + 1e-10) / (union + 1e-10)

def compute_iou(s, g):
    intersection = torch.sum(s * g)
    union = torch.sum(s) + torch.sum(g) - intersection
    return (intersection + 1e-10) / (union + 1e-10)


def evalResult(checkpoint_dir, eval_fig_dir, itr_range):
    [range0, range1] = itr_range
    os.makedirs(eval_fig_dir, exist_ok=True)
    # dataset_name_save = args.dataset_name.replace('/', '_')
    dataset_name_save = args.dataset_name
    with open(f'{eval_fig_dir}/{dataset_name_save}.txt', mode='w') as Note:
        iterations, dices, ious = [],[],[]
        for model_path in sorted(os.listdir(checkpoint_dir)):
            iteration = int(model_path.split('.')[0])
            if range0 <= iteration <= range1 :
                iterations.append(iteration)
                model_path = os.path.join(checkpoint_dir,model_path)
                [FPS_, FPS, Dice, IoU] = test(model_path)
                Note.write("\nmodel:"+model_path.split("/")[-1]+", mean Dice is " + str(Dice) + ', ' + "IoU is " + str(IoU) + ', FPS_:%.2f'%FPS_+' FPS:%.2f'%FPS)
                dices.append(float(Dice))
                ious.append(float(IoU))
                print("\n"+model_path.split("/")[-1]+" mean Dice is " + str(Dice) + ', ' + "IoU is " + str(IoU) + ' FPS_:%.2f'%FPS_+' FPS:%.2f'%FPS)
    # plt.plot(iterations, dices, 'bo--', alpha=0.5, linewidth=1, label='dice')
    # plt.plot(iterations, ious, 'ro--', alpha=0.5, linewidth=1, label='iou')
    # plt.legend()
    # plt.xlabel('iteration')
    # plt.ylabel('score')
    # plt.savefig(f'{eval_fig_dir}/eval.jpg')

def save_mask(pred, file):
    pred = pred.permute(1, 2, 0)
    pred_np = pred.detach().cpu().numpy()
    pred_np = pred_np * 255
    pred_np = pred_np.astype(np.uint8)
    cv2.imwrite(file, pred_np)
    return


def test(checkpoint_file):
    # init
    save_dir = os.path.join(args.exp_dir, args.exp_name, args.save_masks)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # dataloader
    dataset_seg = DatasetSegmentationInfer(args, args.test_seg_dir)
    dataloader_seg = DataLoader(dataset_seg, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # network
    model = Network(args).cuda()
    checkpoint = torch.load(checkpoint_file)
    model_state = checkpoint['model_state_dict']
    model.load_state_dict(model_state)
    model.eval()
    if args.sam == "vit_b":
        SAM_model = sam_model_registry["vit_b"](checkpoint="pretrained/sam_vit_b_01ec64.pth")
    elif args.sam == "vit_h":
        SAM_model = sam_model_registry["vit_h"](checkpoint="pretrained/sam_vit_h_4b8939.pth")
    elif args.sam == "efficient_sam_vitt":
        SAM_model = build_efficient_sam_vitt()
    elif args.sam == "sam2_tiny":
        SAM_model = build_sam2(checkpoint=args.sam2_path_t,config_file="configs/sam2.1/sam2.1_hiera_t.yaml")
    elif args.sam == "sam2_small":
        SAM_model = build_sam2(checkpoint=args.sam2_path_s,config_file="configs/sam2.1/sam2.1_hiera_s.yaml")
    elif args.sam == "sam2_base":
        SAM_model = build_sam2(checkpoint=args.sam2_path_b,config_file="configs/sam2.1/sam2.1_hiera_b+.yaml")
    elif args.sam == "sam2_large":
        SAM_model = build_sam2(checkpoint=args.sam2_path_l,config_file="configs/sam2.1/sam2.1_hiera_l.yaml")
        
    SAM_model.load_state_dict(checkpoint['sam_model_state_dict'], strict=False)
    SAM_model = SAM_model.cuda()
    SAM_model.eval()     
    
    t0 = time.time()
    fps_save, fps_list, frame_num = [], [], 0 # mean fps
    dices,ious = [],[]
    for img, gt, img_paths in tqdm(dataloader_seg):
        img, gt = img.cuda(), gt.cuda()
        gt = torch.stack([transforms.Resize(256)(image) for image in gt])

        with torch.no_grad():
            t1 = time.time()
            image_embeddings, interm_embeddings = SAM_model.image_encoder(img)
            polyp_pred, iou_preds, _ = model(img, image_embeddings, interm_embeddings, multimask_output=False)
            fps_list.append(img.size(0) / (time.time() - t1)) # mean fps

        polyp_pred = polyp_pred >= 0.5
        # visiualize
        # for i, img_path in enumerate(img_paths):
        #     img_name = img_path.split('/')[-1]
        #     polyp_map = polyp_pred[i]
        #     polyp_file = os.path.join(save_dir, img_name.split('.')[0]+'.png')
        #     save_mask(polyp_map, polyp_file)
        t2 = time.time()
        frame_num += img.size(0)
        dices.append(compute_dice(polyp_pred, gt))
        ious.append(compute_iou(polyp_pred, gt))
    fps_save.append(frame_num / (t2 - t0)) # mean save fps
    return [sum(fps_list)/len(fps_list), sum(fps_save)/len(fps_save), "%.4f" %(sum(dices)/len(dices)),"%.4f" %(sum(ious)/len(ious))]

if __name__ == '__main__':
    args.iterations, args.save_iter = int(args.iterations), int(args.save_iter)
    checkpoint_dir = os.path.join(args.exp_dir, args.exp_name, args.chekpoints)
    eval_fig_dir = os.path.join(args.exp_dir, args.exp_name, 'eval_fig')
    evalResult(checkpoint_dir, eval_fig_dir, itr_range=[0,args.iterations]) # 1000,args.iterations
