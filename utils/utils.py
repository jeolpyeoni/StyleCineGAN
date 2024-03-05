import torch
import cv2
import numpy as np
from torchvision import transforms, utils
from PIL import Image


device = torch.device("cuda")


def clip_img(x):
    """Clip stylegan generated image to range(0,1)"""
    img_tmp = x.clone()[0]
    img_tmp = (img_tmp + 1) / 2
    img_tmp = torch.clamp(img_tmp, 0, 1)
    return img_tmp


def gan_inversion(encoder, img, model='fs'):
    
    if model=='fs':
        
        img_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        tensor_img = img_to_tensor(Image.fromarray(img)).unsqueeze(0).to(device)
        
        output = encoder.test(img=tensor_img, return_latent=True)
        feature = output.pop()
        latent = output.pop()
        result = output[1]
        return result, latent, feature    
    
    elif model=='psp':
        ### Resize image to 256X256
        tensor_img = to_tensor(img)
        tensor_input = resize_tensor(tensor_img, (256,256)).cuda()        
        result, latent = encoder(tensor_input, resize=False, randomize_noise=False, return_latents=True)
        return result, latent
    
    
def colorize_mask(mask):
    
    car_32_palette =[ 255,  255,  255,
      238,  229,  102,
      0, 0, 0,
      124,  99 , 34,
      193 , 127,  15,
      106,  177,  21,
      248  ,213 , 42,
      252 , 155,  83,
      220  ,147 , 77,
      99 , 83  , 3,
      116 , 116 , 138,
      63  ,182 , 24,
      200  ,226 , 37,
      225 , 184 , 161,
      233 ,  5  ,219,
      142 , 172  ,248,
      153 , 112 , 146,
      38  ,112 , 254,
      229 , 30  ,141,
      115  ,208 , 131,
      52 , 83  ,84,
      229 , 63 , 110,
      194 , 87 , 125,
      225,  96  ,18,
      73  ,139,  226,
      172 , 143 , 16,
      169 , 101 , 111,
      31 , 102 , 211,
      104 , 131 , 101,
      70  ,168  ,156,
      183 , 242 , 209,
      72  ,184 , 226]
    
    
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(car_32_palette)
    
    return np.array(new_mask.convert('RGB'))
    
        
def predict_mask(opts, basename_input, classifiers, feature_maps_all): 
    
    import scipy.stats
    with torch.no_grad():
        seg_mode_ensemble = []
        for MODEL_NUMBER in range(len(classifiers)):
            classifier = classifiers[MODEL_NUMBER]
            img_seg = classifier(feature_maps_all)
            img_seg = img_seg.squeeze()
            
            img_seg_final = torch.log_softmax(img_seg, dim=1)
            _, img_seg_final = torch.max(img_seg_final, dim=1)
            img_seg_final = img_seg_final.reshape(512, 512, 1)
            img_seg_final = img_seg_final.detach().cpu().numpy()
            seg_mode_ensemble.append(img_seg_final)
            
        img_seg_final = np.concatenate(seg_mode_ensemble, axis=-1)
        img_seg_final = scipy.stats.mode(img_seg_final, 2)[0].reshape(512, 512)
        
        
        ### Mask Optimization Process
        if opts.flownet_mode == "sky":
            mask_final = np.where(img_seg_final==1., 1, 0)
        elif opts.flownet_mode == "fluid":
            mask_final = np.where(img_seg_final==2., 1, 0)
        elif opts.flownet_mode == "sky+fluid":
            mask_final = np.where(img_seg_final==1., 1, 0) + np.where(img_seg_final==2., 1, 0)

        
        ### Save initial mask
        init_mask_image = colorize_mask(mask_final)
        
        
        ### Smoothening
        kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        
        mask_final = 255 * mask_final
        mask_final = mask_final.astype(np.uint8)
        
        mask_final = cv2.dilate(mask_final, kernel_4, iterations=1)
        mask_final = cv2.erode(mask_final, kernel_3, iterations=2)
  
        
        ### Foreground (Landscape) Filtering
        mask_final = 255 - mask_final
        mask_final = mask_final.astype(np.uint8)
        _, contours, hierarchy = cv2.findContours(mask_final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                if cv2.contourArea(cnt) < 512*512*0.1:
                    cv2.drawContours(mask_final, [cnt], 0, (255), -1)

                
        ### Background (Sky+Fluid) Filtering
        mask_final = 255 - mask_final
        _, contours, hierarchy = cv2.findContours(mask_final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                if cv2.contourArea(cnt) < 512*512*0.1:
                    cv2.drawContours(mask_final, [cnt], 0, (255), -1)
        
        
        ### Final Refinement   
        mask_final = mask_final / 255
        
        mask_sum = np.sum(mask_final)
        
        if mask_sum < 512*512*0.1:
            mask_final = np.ones_like(mask_final).astype(np.uint8)
        else:
            mask_final = mask_final.astype(np.uint8)
            
        mask_image = colorize_mask(mask_final)      
        mask_final = torch.from_numpy(mask_final).unsqueeze(0).to(feature_maps_all.device)
        
    return mask_final, init_mask_image, mask_image

