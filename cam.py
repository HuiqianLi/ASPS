import cv2
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from segment_anything import sam_model_registry
from sam2.build_sam import build_sam2
from model.Network import Network
from option import args

def save_mask(pred, file):
    pred = pred.permute(1, 2, 0)
    pred_np = pred.detach().cpu().numpy()
    pred_np = pred_np * 255
    pred_np = pred_np.astype(np.uint8)
    cv2.imwrite(file, pred_np)
    return

class CAMGenerator_SAM:
    def __init__(self, model):
        self.model = model.cuda()
        self.gradients = None
        self.activations = None

        # 注册钩子获取sam_feature
        # image_encoder.neck.3
        model.image_encoder.blocks[31].mlp.lin2.register_forward_hook(self._activation_hook)
        model.image_encoder.blocks[31].mlp.lin2.register_backward_hook(self._gradient_hook)

    def _activation_hook(self, module, input, output):
        self.activations = output.detach()

    def _gradient_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        :param input_tensor: 输入图像 (B,C,H,W)
        :param target_class: 目标类别通道 (默认为分割主通道)
        """
        input_tensor.requires_grad_(True)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 前向传播
        output, _ = self.model.image_encoder(input_tensor)
        
        # 自动选择目标通道
        if target_class is None:
            target = output.max(dim=3)[0].sum()  # 选择置信度最高通道
        else:
            target = output[:, target_class, :, :].sum()
            
        # 反向传播获取梯度
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # 计算通道权重
        pooled_grads = torch.mean(self.gradients, dim=[0,1,2])
        weighted_activations = pooled_grads[None, None, :] * self.activations
        
        # 生成CAM
        cam = weighted_activations.sum(dim=3).squeeze()
        # cam = torch.relu(cam)  # ReLU增强可视化效果
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.detach().cpu().numpy()
    
class CAMGenerator:
    def __init__(self, model):
        self.model = model.cuda()
        self.gradients = None
        self.activations = None

        # # 注册钩子获取cnn_feature
        # model.encoder.model.block4[2].mlp.fc2.register_forward_hook(self._activation_hook)
        # model.encoder.model.block4[2].mlp.fc2.register_backward_hook(self._gradient_hook)

        # # 注册钩子获取fusion_feature
        # model.mask_decoder.embedding_maskfeature[3].register_forward_hook(self._activation_hook)
        # model.mask_decoder.embedding_maskfeature[3].register_backward_hook(self._gradient_hook)

        # 注册钩子获取 img_pe
        model.mask_decoder.cnn_modify[2].register_forward_hook(self._activation_hook)
        model.mask_decoder.cnn_modify[2].register_backward_hook(self._gradient_hook)

    def _activation_hook(self, module, input, output):
        self.activations = output.detach()

    def _gradient_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        :param input_tensor: 输入图像 (B,C,H,W)
        :param target_class: 目标类别通道 (默认为分割主通道)
        """
        input_tensor.requires_grad_(True)
        for param in self.model.parameters():
            param.requires_grad = True
            
        # 前向传播
        output, _, _ = self.model(input_tensor, image_embeddings, interm_embeddings, multimask_output=False)
        
        # 自动选择目标通道
        if target_class is None:
            target = output.max(dim=1)[0].sum()  # 选择置信度最高通道
        else:
            target = output[:, target_class, :, :].sum()
            
        # 反向传播获取梯度
        self.model.zero_grad()
        target.backward(retain_graph=True)
        
        # 计算通道权重
        pooled_grads = torch.mean(self.gradients, dim=[0,2,3])
        weighted_activations = pooled_grads[:, None, None] * self.activations
        
        # 生成CAM
        cam = weighted_activations.sum(dim=1).squeeze()
        cam = torch.relu(cam)  # ReLU增强可视化效果
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam.detach().cpu().numpy()

def visualize_cam(model, image_tensor, save_path, model_type=None):
    # 初始化生成器
    if model_type=="sam":
        cam_generator = CAMGenerator_SAM(model)
    else:
        cam_generator = CAMGenerator(model)
    
    # 获取CAM热力图
    cam = cam_generator.generate(image_tensor)
    
    # 可视化配置
    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    img_np = image_tensor[0].permute(1,2,0).cpu().detach().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    # 直接保存 在原图上的叠加效果
    img_np = cv2.resize(img_np, (224,224)) # (224,224) means the size of the raw image
    cam = cv2.resize(cam, (224,224))
    img_np = np.uint8(255 * img_np)
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # use the 'jet' colormap to apply on the heatmap
    alpha = 0.5  # the transparency of the heatmap
    overlay = cv2.addWeighted(img_np, alpha, cam, 1 - alpha, 0)
    cv2.imwrite(save_path, overlay)

if __name__ == '__main__':
    checkpoint_file = "/data/dataset/lhq/code/SAM/ASPS/exp_dir/h_l/H-L-0016400.pth"
    # image_path = "/data/dataset/lhq/code/SAM/ASPS/sample/image.jpg"
    image_path = "/data/dataset/lhq/data/polyp_seg_PraNet/TrainDataset/images/11.png"
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
    del input_tensor

    # gt = Image.open("exp_dir/gt.png").convert('L')
    # gt = transforms.Resize((256, 256))(gt)
    # gt = transforms.ToTensor()(gt).unsqueeze(0).cuda()

    if args.sam == "vit_b":
        model_encoder = sam_model_registry["vit_b"](checkpoint="pretrained/sam_vit_b_01ec64.pth")
    elif args.sam == "vit_h":
        model_encoder = sam_model_registry["vit_h"](checkpoint="pretrained/sam_vit_h_4b8939.pth")
    elif args.sam == "sam2_large":
        model_encoder = build_sam2(checkpoint=args.sam2_path_l,config_file="configs/sam2.1/sam2.1_hiera_l.yaml")
    model_encoder = model_encoder.cuda()
    model_encoder.eval()
    model = Network(args).cuda()
    checkpoint = torch.load(checkpoint_file)
    model_state = checkpoint['model_state_dict']
    model.load_state_dict(model_state)   # , strict=False)
    model_encoder.load_state_dict(checkpoint['sam_model_state_dict'], strict=False)
    model.eval()

    images = torch.stack([transforms.Resize(1024)(image) for image in input_batch])
    with torch.no_grad():
        image_embeddings, interm_embeddings = model_encoder.image_encoder(images)
    # masks, iou_preds, uncertainty_p = model(images, image_embeddings, interm_embeddings, multimask_output=False)

    # 执行可视化
    visualize_cam(model, images, "cam_visualization.png")
    # visualize_cam(model_encoder, images, "cam_visualization.png", model_type="sam")