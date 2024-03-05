
import torch
import torch.nn as nn
import mediapy as mp
from PIL import Image


device = "cuda"


def read_image(img_dir, dim=1024, is_square=True, return_tensor=True):
    
    np_img = mp.read_image(img_dir)
    
    # crop to square
    if is_square:
        np_img = crop_image(np_img)
    
    # resize
    np_img = mp.resize_image(np_img, (dim, dim))

    # to tensor
    if return_tensor:
        return to_tensor(np_img)
    return np_img


def resize_tensor(x, size):
    return torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)


def to_tensor(x, norm=True):
    x = torch.from_numpy(x).float()
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0)
    if norm:
        x = x / 127.5 - 1
    return x


def to_numpy(x, norm=True):
    if norm:
        x = (x + 1) / 2
    x = x.permute(0, 2, 3, 1)
    x = x.cpu().detach().numpy()
    return x


def to_im(var):
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def crop_image(x):
    # crop image to square
    H, W, _ = x.shape
    
    if H != W:

        if H < W:

            pos1 = (W - H) / 2
            pos2 = (W + H) / 2

            pos1 = int(pos1)
            pos2 = int(pos2)
            
            x = x[:,pos1:pos2,:]

        elif H > W:

            pos1 = (H - W) / 2
            pos2 = (H + W) / 2
            
            pos1 = int(pos1)
            pos2 = int(pos2)

            x = x[pos1:pos2,:]
    return x


class Interpolate(nn.Module):
    def __init__(self, size, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if self.align_corners:
            x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        else:
            x = self.interp(x, size=self.size, mode=self.mode)
        return x


def make_sg2_features(sg2, latent):

    out_res = 512
    
    ### Setup feature upsamplers
    mode = 'bilinear'
    upsamplers = [nn.Upsample(scale_factor=out_res / 4, mode=mode),
                  nn.Upsample(scale_factor=out_res / 8, mode=mode),
                  nn.Upsample(scale_factor=out_res / 8, mode=mode),
                  nn.Upsample(scale_factor=out_res / 16, mode=mode),
                  nn.Upsample(scale_factor=out_res / 16, mode=mode),
                  nn.Upsample(scale_factor=out_res / 32, mode=mode),
                  nn.Upsample(scale_factor=out_res / 32, mode=mode),
                  nn.Upsample(scale_factor=out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, mode=mode)]

    upsamplers.append(Interpolate(512, mode))
    upsamplers.append(Interpolate(512, mode))
    
    
    with torch.no_grad():
        
        ### Make sg2 features
        _, affine_layers = sg2.module.return_forward([latent],
                                              input_is_latent=True,
                                              randomize_noise=False,
                                              return_latents=False,
                                              return_features=True
                                             )

        feature_maps = []
        for i in range(len(affine_layers)):
            feature_maps.append(upsamplers[i](affine_layers[i]))

        feature_maps_all = torch.cat(feature_maps, dim=1)
        feature_maps_all = feature_maps_all.permute(0, 2, 3, 1)
        feature_maps_all = feature_maps_all.reshape(-1, 5568).float().cuda()

        return feature_maps_all
    
    
def make_sg2_features_fs(sg2, latent, feature):

    out_res = 512
    
    ### Setup feature upsamplers
    mode = 'bilinear'
    upsamplers = [nn.Upsample(scale_factor=out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, mode=mode),
                  nn.Upsample(scale_factor=out_res / 512, mode=mode)]

    upsamplers.append(Interpolate(512, mode))
    upsamplers.append(Interpolate(512, mode))
    
    
    with torch.no_grad():
        
        sg2.to(latent.device)
        
        ### Make sg2 features
        _, _, _, affine_layers = sg2.module.feature_forward([latent],
                                                     feature,
                                                     input_is_latent=True,
                                                     randomize_noise=False,
                                                     return_latents=True
                                                    )

        feature_maps = []
        for i in range(len(affine_layers)):
            feature_maps.append(upsamplers[i](affine_layers[i]))

        feature_maps_all = torch.cat(feature_maps, dim=1)
        feature_maps_all = feature_maps_all.permute(0, 2, 3, 1)
        feature_maps_all = feature_maps_all.reshape(-1, 1472).float().cuda()

        return feature_maps_all

