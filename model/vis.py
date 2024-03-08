import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from einops import rearrange

from model.Network import Network
from option import args

def show_heatmap(data,output_jpg_name,raw_image):
    heatmap = np.mean(data, axis=0)  # calculate the mean of feature maps along the channel dimension
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)  # normalize the heatmap

    heatmap = cv2.resize(heatmap, (224,224)) # (224,224) means the size of the raw image
    raw_image = cv2.resize(raw_image, (224,224))

    heatmap = np.uint8(255 * heatmap)
    heatmap_colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # use the 'jet' colormap to apply on the heatmap
    alpha = 0.5  # the transparency of the heatmap
    overlay = cv2.addWeighted(raw_image, alpha, heatmap_colormap, 1 - alpha, 0)

    cv2.imwrite(output_jpg_name, overlay)

def show_heatmap_n(feature,output_jpg_name,raw_image):
    data = feature.cpu().detach().numpy()[0]
    show_heatmap(data,output_jpg_name,raw_image)

def show_heatmap_t(feature,output_jpg_name,raw_image):
    feature = feature[0].permute(2, 0, 1)
    data = feature.cpu().detach().numpy()
    show_heatmap(data,output_jpg_name,raw_image)

def show_amplitude(feature, output_jpg_name):
    # fourier transform
    feature = feature.cpu().detach().numpy()
    feature = np.squeeze(np.sum(feature, axis=1))  # sum over the channel dimension, grayscale image
    freq = np.fft.fft2(feature, axes=(0, 1))
    freq = np.fft.fftshift(freq)

    # calculate amplitude spectrum
    amplitude_spectrum = np.abs(freq)
    amplitude_spectrum = np.log(1 + amplitude_spectrum)  # log scaling
    amplitude_spectrum = (amplitude_spectrum - np.min(amplitude_spectrum)) / (np.max(amplitude_spectrum) - np.min(amplitude_spectrum)) * 255
    amplitude_spectrum = amplitude_spectrum.astype('uint8').copy()

    # apply colormap
    amplitude_spectrum = cv2.applyColorMap(amplitude_spectrum, cv2.COLORMAP_BONE)
    cv2.imwrite(output_jpg_name, amplitude_spectrum)
    

def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f

def shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2, 3))

    
def show_latent(latents, labels, output_jpg_name):
    # Fourier transform feature maps
    fourier_latents = []
    for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
        latent = latent.cpu()
        
        if len(latent.shape) == 3:  # for ViT
            b, n, c = latent.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        elif len(latent.shape) == 4:  # for CNN
            b, c, h, w = latent.shape
        else:
            raise Exception("shape: %s" % str(latent.shape))
        latent = fourier(latent)
        latent = shift(latent).mean(dim=(0, 1))
        latent = latent.diag()[int(h/2):]  # only use the half-diagonal components
        latent = latent - latent[0]  # visualize 'relative' log amplitudes 
                                    # (i.e., low-freq amp - high freq amp)
        fourier_latents.append(latent)
        
    # A. Plot Fig 2a: "Relative log amplitudes of Fourier transformed feature maps"
    fig, ax1 = plt.subplots(1, 1, figsize=(5.5, 4), dpi=150)
    for i, latent in enumerate(fourier_latents):
        freq = np.linspace(0, 1, len(latent))
        latent_np = latent.detach().numpy()
        ax1.plot(freq, latent_np, color=cm.plasma_r(i / len(fourier_latents)), label=labels[i])
        
    plt.legend()
    ax1.set_xlim(left=0, right=1)

    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("$\Delta$ Log amplitude")

    from matplotlib.ticker import FormatStrFormatter
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))

    # save png
    plt.savefig(output_jpg_name)
    # save svg
    # output_svg_name = output_jpg_name.replace('png', 'svg')
    # plt.savefig(output_svg_name, format='svg')

if __name__ == '__main__':
    features = torch.randn(2, 256, 10, 10)
    checkpoint_file = ''
    raw_img = cv2.imread('image0001.jpg')
    out_name = 'heatmap.png'

        # import cv2
        # from .vis import show_heatmap_n, show_heatmap_t
        # raw_img = cv2.imread('image0001.jpg')
        # out_name = 'exp_dir/dense_prompt_tokens.png'
        # print(self.dense_prompt_tokens.shape)
        # show_heatmap_n(self.dense_prompt_tokens, out_name, raw_img)