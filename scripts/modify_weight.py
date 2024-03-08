import torch

# To modify the pretrained weights, we need to load the weights and then save them again. [for MSCAN]
# load the pretrained weights
pretrained_weights = torch.load('./pretrained/segnext_large_512x512_ade_160k.pth')

# delete "meta" 和 "optimizer" keys
if "meta" in pretrained_weights:
    del pretrained_weights["meta"]
if "optimizer" in pretrained_weights:
    del pretrained_weights["optimizer"]

# bulid new state_dict
new_state_dict = {}

for key, value in pretrained_weights['state_dict'].items():
    if "decode" in key:
        continue
    if key.startswith("backbone."):
        new_key = key.replace("backbone.", "")
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# save the new state_dict
torch.save(new_state_dict, './pretrained/mscan_l.pth')