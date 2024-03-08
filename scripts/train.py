import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore")

from option import args
from segment_anything import sam_model_registry
from dataset.Segmentation import DatasetSegmentation, sample_data
from loss.segmentation import dice_loss
from model.Network import Network

class Trainer():
    def __init__(self):
        super(Trainer, self).__init__()
        self.dice_loss = dice_loss
        self.ce_loss = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()
    
    def seg_loss(self, pred, gt):
        loss = self.ce_loss(pred, gt) + 0.5 * self.dice_loss(pred, gt) + self.mse_loss(pred, gt)
        return loss
    
    def train(self):
        # init
        checkpoint_dir = os.path.join(args.exp_dir, args.exp_name, args.chekpoints)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # dataloader
        dataset_polyp_seg = DatasetSegmentation(args, args.polyp_dir)
        dataloader_polyp_seg = DataLoader(dataset_polyp_seg, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True)
        dataloader_polyp_seg = sample_data(dataloader_polyp_seg)

        # network  
        model = Network(args)
        model = model.cuda()
        if args.sam == "vit_b":
            SAM_model = sam_model_registry["vit_b"](checkpoint="pretrained/sam_vit_b_01ec64.pth")
        else:
            SAM_model = sam_model_registry["vit_h"](checkpoint="pretrained/sam_vit_h_4b8939.pth")
        SAM_model = SAM_model.cuda()
        SAM_model.eval()

        # optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for param in SAM_model.parameters():
            param.requires_grad = False
        #  训练LN层
        for name, param in SAM_model.named_parameters():
            if 'image_encoder.neck.3'  in name or 'image_encoder.neck.1' in name:
            # if 'image_encoder.neck.3'  in name or 'image_encoder.neck.1' in name or 'norm' in name:
            # if 'norm' in name:
                param.requires_grad = True

        pbar = tqdm(range(1, args.iterations+1))
        lmbda = 0.1

        for itr in pbar:
            model.train()

            # train polyp segmentation
            img, gt = next(dataloader_polyp_seg)
            img, gt = img.cuda(), gt.cuda()
            gt = torch.stack([transforms.Resize(256)(image) for image in gt])

            image_embeddings, interm_embeddings = SAM_model.image_encoder(img)
            pred, iou_pred, uncertainty_p = model(img, image_embeddings, interm_embeddings, multimask_output=False)

            # 像素不确定性+图像不确定性
            ones = torch.ones(iou_pred.shape).cuda()
            confidence = (iou_pred + (ones - uncertainty_p.mean(dim=(2, 3)))) / 2
            # Make sure we don't have any numerical instability
            eps = 1e-12
            pred = torch.clamp(pred, 0. + eps, 1. - eps)
            confidence = torch.clamp(confidence, 0. + eps, 1. - eps)

            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).cuda()
            # 提示的权重由像素不确定性决定
            pred_new = torch.zeros(pred.shape).cuda() 
            for i in range(pred.shape[0]):
                pred_new[i] = pred[i] * (1 - uncertainty_p[i]) + gt[i] * uncertainty_p[i] if b[i] else pred[i]
                # P' = c * P + (1-c) * Y
                # pred_new[i] = confidence[i] * pred[i] + (1 - confidence[i]) * gt[i] if b[i] else pred[i]
            
            optimizer.zero_grad()
            confidence_loss = torch.mean(-torch.log(confidence))
            loss = self.seg_loss(pred_new, gt)  + (lmbda * confidence_loss)

            if args.budget > confidence_loss.item():
                lmbda = lmbda / 1.01
            elif args.budget <= confidence_loss.item():
                lmbda = lmbda / 0.99
                
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clipping)
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            # if itr % args.save_iter == 0 or itr == args.iterations:
            if itr >= 40000 and (itr % args.save_iter == 0 or itr == args.iterations):
                save_file = os.path.join(checkpoint_dir, f'{str(itr).zfill(7)}.pth')
                torch.save(model.state_dict(), save_file)
                print('checkpoint saved at: ', save_file)
    


if __name__ == '__main__':
    Trainer = Trainer()
    Trainer.train()
