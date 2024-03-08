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
from dataset.Segmentation import DatasetSegmentationInfer
from model.Network import Network
from segment_anything import sam_model_registry

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
    dataset_name_save = args.dataset_name.replace('/', '_')
    with open(f'{eval_fig_dir}/preds_seg_{dataset_name_save}.txt', mode='w') as Note:
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
    model.load_state_dict(torch.load(checkpoint_file))
    model.eval()
    if args.sam == "vit_b":
        SAM_model = sam_model_registry["vit_b"](checkpoint="pretrained/sam_vit_b_01ec64.pth")
    else:
        SAM_model = sam_model_registry["vit_h"](checkpoint="pretrained/sam_vit_h_4b8939.pth")
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
            # # 执行白化
            # mean, std = img.mean(dim=(2, 3)), img.std(dim=(2, 3))
            # whitened_image = (img - mean[:, :, None, None]) / std[:, :, None, None]
            image_embeddings, interm_embeddings = SAM_model.image_encoder(img)
            polyp_pred, iou_preds, _ = model(img, image_embeddings, interm_embeddings, multimask_output=False)
            fps_list.append(img.size(0) / (time.time() - t1)) # mean fps

        polyp_pred = polyp_pred >= 0.5
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
    checkpoint_dir = os.path.join(args.exp_dir, args.exp_name, args.chekpoints)
    eval_fig_dir = os.path.join(args.exp_dir, args.exp_name, 'eval_fig')
    evalResult(checkpoint_dir, eval_fig_dir, itr_range=[0,args.iterations]) # 1000,args.iterations
