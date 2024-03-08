import argparse
parser = argparse.ArgumentParser(description='MSACAN_SAM')


# common
parser.add_argument('--exp_name', default='sample')
parser.add_argument('--iterations', type=int, default=400*40, help='the number of iterations for training')
parser.add_argument('--seed', type=int, default=0, help='total seed')
parser.add_argument('--lr', type=float, default=1e-5)   # 5e-5
parser.add_argument('--weight_decay', type=float, default=1e-4) # 1e-4
parser.add_argument('--clipping', type=float, default=2, help='Gradient clipping')
parser.add_argument('--exp_dir', default='exp_dir')
parser.add_argument('--chekpoints', default='chekpoints')
parser.add_argument('--save_masks', default='save_masks')
parser.add_argument('--save_iter', default=400) # 5000

# backbone
parser.add_argument('--mscan_checkpoint', type=str, default='pretrained/mscan_l.pth')
parser.add_argument('--mscan', type=str, default='large')
parser.add_argument('--sam', type=str, default='efficient_sam_vitt', help='vit_b or vit_h or efficient_sam_vitt')

# segmentation
parser.add_argument('--RFB_aggregated_channel', type=int, nargs='*', default=[32, 64, 128])
parser.add_argument('--denoise', type=float, default=0.93, help='Denoising background ratio')
parser.add_argument('--budget', type=float, default=0.3, metavar='N', help='the budget for how often the network can get hints')

# dataloader
parser.add_argument('--polyp_dir', default='data/sample')
parser.add_argument('--dataset_name', default='CVC-300')
parser.add_argument('--test_seg_dir', default='data/sample')
parser.add_argument('--image_size', type=int, default=1024, help='image size used during training')
parser.add_argument('--ver', type=int, default=2, help='type of transform')
parser.add_argument('--batch_size', type=int, default=4, help='batch size in each mini-batch') # 4
parser.add_argument('--num_workers', type=int, default=4, help='number of workers used in data loader')

args = parser.parse_args()
