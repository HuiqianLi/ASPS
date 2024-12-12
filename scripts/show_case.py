from PIL import Image
import os
from tqdm import tqdm

# 文件夹路径
Images = "/root/autodl-tmp/polyp_seg/TestDataset/ETIS-LaribPolypDB/images/"
GT = "/root/autodl-tmp/polyp_seg/TestDataset/ETIS-LaribPolypDB/masks/"
PraNet = "/root/autodl-tmp/code_SAM/00-PraNet/ETIS-LaribPolypDB"
SAMAdapter = "/root/autodl-tmp/code_SAM/01-SAMAdapter/exp_dir/0225_B/save_masks/"
MedSAM = "/root/autodl-tmp/code_SAM/02-MedSAM/exp_dir/0227-b/save_masks/"
SAMUS = "/root/autodl-tmp/code_SAM/02-SAMUS/exp_dir/0229_b/save_masks/"
ours = "/root/autodl-tmp/code_SAM/03-SAM-Alation/v4_EFA_PGA/exp_dir/0302_B_L/save_masks/"
target = "/root/autodl-tmp/code_SAM/res/ETIS-LaribPolypDB"

folder_paths = [Images, GT, PraNet, SAMAdapter, MedSAM, SAMUS, ours]
os.makedirs(target, exist_ok=True)

for image_name in tqdm(os.listdir(PraNet)):
    # 存储每个文件夹下的图片路径
    image_paths = []

    # 遍历文件夹，获取图片路径
    for folder_path in folder_paths:
        folder_image_path = os.path.join(folder_path, image_name)
        if os.path.exists(folder_image_path):
            image_paths.append(folder_image_path)
        if os.path.exists(folder_image_path.replace('png', 'jpg')):
            image_paths.append(folder_image_path.replace('png', 'jpg'))

    # 创建一个新的图像对象，用于合并图片
    result_image = Image.new('RGB', (len(image_paths) * 256, 256))

    # 按顺序加载并合并图片
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        image = image.resize((256, 256))
        result_image.paste(image, (i * 256, 0))

    # 保存结果图像
    save_path = os.path.join(target, image_name)
    result_image.save(save_path)