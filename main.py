import torch
from tqdm import tqdm
import mediapy as mp
import numpy as np
import os
import cv2

from PIL import Image

import warnings
warnings.filterwarnings(action='ignore')


device = torch.device('cuda')


from option import Options
from utils.model_utils import load_encoder, load_stylegan2
from utils.ip_utils import read_image, resize_tensor, to_numpy, to_tensor, make_sg2_features, make_sg2_features_fs
from utils.utils import clip_img, gan_inversion, predict_mask
from utils.flow_utils import z_score_filtering, flow2img
from utils.cinemagraph_utils import feature_inpaint, resize_flow, resize_feature



if __name__ == "__main__":
    
    # load options -----------------------------------------------------------------------------------
    opts = Options().parse()
    opts.device = device
    
    # make save directory ---------------------------------------------------------------------------
    os.makedirs(f"{opts.save_dir}", exist_ok=True)
    
    
    # load models ------------------------------------------------------------------------------------
    sg2      = load_stylegan2(opts.sg2_ckpt)
    encoder  = load_encoder(opts.encoder_ckpt, encoder_type=opts.encoder_type, recon_idx=opts.recon_feature_idx).to(device)
    

    # read images ------------------------------------------------------------------------------------
    basename_input = (opts.img_path).split("/")[-1]
    torch_input = read_image(f"{opts.img_path}/{basename_input}.png", dim=1024, is_square=True, return_tensor=False)
    
    if opts.style_path:
        torch_style = read_image(opts.style_path, is_square=True, return_tensor=True)
        basename_style = os.path.basename(opts.style_path).split('.')[0]
    
    
    # invert the image ------------------------------------------------------------------------------
    with torch.no_grad():
        tensor_recon, latent, feature = gan_inversion(encoder, torch_input, model=opts.encoder_type)
        
    if opts.style_path:
        mean_latent = sg2.mean_latent(10000)
        mean_latent = mean_latent.detach()
        torch_style = to_tensor(torch_style)
        
        # invert the style image via optimization
        from optimize_latent import OptimizeLatent
        optim_latents = OptimizeLatent(opts, sg2, threshold=opts.optim_threshold)
        tensor_recon_style, latent_style = optim_latents.optimize_latent(torch_style, mean_latent, step=1500,
                                                                         initial_lr=opts.initial_lr,
                                                                         optim_params='latent')
    torch_input = to_tensor(torch_input)
        
    # visualize inversion results -------------------------------------------------------------------
    if opts.vis == "True":
        img_list = [torch_input, tensor_recon]
        if opts.style_path:
            img_list += [torch_style, tensor_recon_style]
        
        img_list = [to_numpy(resize_tensor(img, [512,512]))[0] for img in img_list]
        img_list = np.concatenate(img_list, axis=1)
        mp.write_image(f"{opts.save_dir}/{basename_input}_recon.png", img_list)
    
    
    # load flow ----------------------------------------------------------------------------------
    print(f"\n>>> Loading Flow...")
    flow = np.load(f"{opts.img_path}/{basename_input}_motion.npy")
    flow = torch.from_numpy(flow).to(device)
    print(">>> Done -------------------------- \n")
        
    
    # load mask ----------------------------------------------------------------------------------
    print("\n>>> Loading Mask...")
    mask = mp.read_image(f"{opts.img_path}/{basename_input}_mask.png")
    mask = mp.resize_image(mask, (512, 512))[:,:,0] / 255
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)
    flow *= mask
    print(">>> Done -------------------------- \n")
        
    
    # visualize optical flow---------------------------------------------------------------
    if opts.vis == "True":
        flows = [flow2img(flow[0].permute(1, 2, 0).cpu().detach().numpy())]
        flows = np.concatenate(flows, axis=1)
        mp.write_image(f"{opts.save_dir}/{basename_input}_flow.png", flows)
    
    
    # generate cinemagraph ---------------------------------------------------------------------------   
    print("\n>>> Generating Cinemagraph...")
    
    # get alphas
    alphas = torch.ones(opts.n_frames, 1).to(device)
    if opts.style_interp:
        alphas = get_alphas(opts.n_frames, opts.style_extrapolate_scale)

    frames = []
    with torch.no_grad():
        latents = []

        # get all latents
        if opts.style_interp:
            for alpha in alphas:
                input_latent = latent_style * alpha +  latent * (1 - alpha)
                latents.append(input_latent)
        else:
            for alpha in alphas:
                latents.append(latent)
    
        # generate frames
        pbar = tqdm(total=len(latents))
        for idx, input_latent in enumerate(latents):    
            result, _ = sg2.module.warp_blend_feature(styles=[input_latent],
                                                        feature=feature,
                                                        idx=idx,
                                                        n_frames=opts.n_frames,
                                                        flow=flow,
                                                        mode=opts.mode,
                                                        Z=None,
                                                        recon_feature_idx=opts.recon_feature_idx,
                                                        warp_feature_idx=opts.warp_feature_idx,
                                                        input_is_latent=True,
                                                        return_latents=True,
                                                        randomize_noise=False,
                                                        is_random=False
                                              )
            
            up_mask = resize_feature(mask.float(), 1024)
            up_flow = resize_flow(flow, 1024)
            
            if opts.image_inpainting:
                result = feature_inpaint(result, up_flow, idx, opts.n_frames)
                
            if not opts.no_image_composit:
                result = result*up_mask + torch_input.cuda() * (1 - up_mask)
            
            result = to_numpy(result)[0]
            frames.append(np.array(result))
            
            pbar.update(1)
        pbar.close()
            
        
        # save video
        if opts.style_path:
            vid_name = f"{opts.save_dir}/{basename_input}_{basename_style}.mp4"
        else:
            vid_name = f"{opts.save_dir}/{basename_input}.mp4"\
            
        mp.write_video(vid_name, frames, fps=30)
        print(">>> Done -------------------------- \n")
        