import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple
import torchvision.transforms as transforms

from .mask_decoder import MaskDecoder
from .CNN_encoder import Mscan_Encoder


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.encoder = Mscan_Encoder(args)

        if args.mscan == 'tiny':
            mscan_dim = 256
        else:
            mscan_dim = 512
        self.mask_decoder = MaskDecoder(model_type=args.sam, transformer_dim=256, cnn_dim=mscan_dim)
        self.cnn_image_size = 320

    def forward(
        self, batched_input, image_embeddings, interm_embeddings, multimask_output=False
    ):
        scaled_images = torch.stack(
            [transforms.Resize(self.cnn_image_size)(image) for image in batched_input]
        )
        features = self.encoder(scaled_images)
        cnn_feature = features[4]

        masks, iou_pred = self.mask_decoder(
            cnn_feature=cnn_feature,
            image_embeddings=image_embeddings,
            interm_embeddings=interm_embeddings,
            multimask_output=multimask_output,
        )
        # U_p, higher score means higher uncertainty
        uncertainty_p = 1 - torch.sigmoid(torch.abs(masks))
        return torch.sigmoid(masks), torch.sigmoid(iou_pred), uncertainty_p


    # from .vis import show_latent
    # out_name = 'exp_dir/fourier.png'
    # interm_1 = interm_embeddings[1].permute(0, 3, 1, 2)
    # interm_2 = interm_embeddings[2].permute(0, 3, 1, 2)
    # interm_3 = interm_embeddings[3].permute(0, 3, 1, 2)
    # latents = [features[2], features[3], features[4], interm_1, interm_2, interm_3]
    # labels = ['cnn_2', 'cnn_3', 'cnn_4', 'vit_1', 'vit_2', 'vit_3']
    # show_latent(latents, labels, out_name)
    # from .vis import show_amplitude
    # output_jpg_name = 'exp_dir/cnn.png'
    # show_amplitude(features[1], output_jpg_name)
    # from .vis import show_amplitude
    # output_jpg_name = 'exp_dir/vit.png'
    # show_amplitude(image_embeddings, output_jpg_name)
    # output_jpg_name = 'exp_dir/cnn.png'
    # show_amplitude(features[2], output_jpg_name)

if __name__ == "__main__":
    import torch
    from option import args

    img = torch.randn(4, 3, 1024, 1024).cuda()
    model = Network(args).cuda()

    image_embeddings = torch.randn(4, 256, 64, 64).cuda()
    interm_embeddings = torch.randn(4, 256, 64, 64).cuda()
    while True:
        out = model(img, image_embeddings, interm_embeddings)
        print(out[0].size())
