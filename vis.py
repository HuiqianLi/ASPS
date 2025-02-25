import cv2
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

from segment_anything import sam_model_registry
from model.Network import Network
from option import args

def save_mask(pred, file):
    pred = pred.permute(1, 2, 0)
    pred_np = pred.detach().cpu().numpy()
    pred_np = pred_np * 255
    pred_np = pred_np.astype(np.uint8)
    cv2.imwrite(file, pred_np)
    return

if __name__ == '__main__':
    checkpoint_file = "chekpoints/0015200.pth"
    image_path = "exp_dir/10.jpg"
    raw_img = cv2.imread(image_path)

    # load image
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.cuda()

    # gt = Image.open("exp_dir/gt.png").convert('L')
    # gt = transforms.Resize((256, 256))(gt)
    # gt = transforms.ToTensor()(gt).unsqueeze(0).cuda()

    with torch.no_grad():

        if args.sam == "vit_b":
            model_encoder = sam_model_registry["vit_b"](checkpoint="pretrained/sam_vit_b_01ec64.pth")
        elif args.sam == "vit_h":
            model_encoder = sam_model_registry["vit_h"](checkpoint="pretrained/sam_vit_h_4b8939.pth")
        elif args.sam == "sam2_large":
            model_encoder = build_sam2(checkpoint=args.sam2_path_l,config_file="configs/sam2.1/sam2.1_hiera_l.yaml")
        model_encoder = model_encoder.cuda()
        model_encoder.eval()
        model = Network(args).cuda()
        # import torch.nn as nn
        # model = nn.DataParallel(model)  # multi-GPU
        checkpoint = torch.load(checkpoint_file)
        model_state = checkpoint['model_state_dict']
        model.load_state_dict(model_state)
        model_encoder.load_state_dict(checkpoint['sam_model_state_dict'], strict=False)
        model.eval()

        images = torch.stack([transforms.Resize(1024)(image) for image in input_batch])
        image_embeddings, interm_embeddings = model_encoder.image_encoder(images)
        masks, iou_preds, uncertainty_p = model(images, image_embeddings, interm_embeddings, multimask_output=False)


    # # U_p + U_i
    # ones = torch.ones(iou_preds.shape).cuda()
    # confidence = (iou_preds + (ones - uncertainty_p.mean(dim=(2, 3)))) / 2
    # # Make sure we don't have any numerical instability
    # eps = 1e-12
    # pred = torch.clamp(masks, 0. + eps, 1. - eps)
    # confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

    # pred_new = torch.zeros(pred.shape).cuda() 
    # for i in range(pred.shape[0]):
    #     pred_new[i] = confidence[i] * pred[i] + (1 - confidence[i]) * gt[i]

    # # polyp_pred = pred_new >= 0.5
    polyp_pred = masks >= 0.5
    img_name = image_path.split('/')[-1]
    polyp_file = os.path.join('exp_dir/', img_name.split('.')[0]+'.png')
    save_mask(polyp_pred[0], polyp_file)
