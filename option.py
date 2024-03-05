import argparse


#### inference opts
class Options():
        
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Arguments for cinemagraph generation")


        ### set models
        self.parser.add_argument("--encoder_type",  type=str, default="fs", help="[fs, psp]")
        self.parser.add_argument("--encoder_ckpt",  type=str, default="./pretrained_models", help="Feature-Style encoder chekpoint directory")
        self.parser.add_argument("--sg2_ckpt",      type=str, default="./pretrained_models/stylegan2-pytorch/sg2-lhq-1024.pt", help="stylegan2 generator checkpoint directory")
        self.parser.add_argument("--datasetgan_dir", type=str, default="./pretrained_models/datasetGAN", help="datasetGAN checkpoint directory")
        self.parser.add_argument("--eulerian_dir", type=str, default="./pretrained_models/epoch-20-feature_encoder.pth", help="eulerian feature encoder directory")
        self.parser.add_argument("--flownet_dir",   type=str, default="./pretrained_models", help="flownet checkpoint directory")
        self.parser.add_argument("--flownet_mode",  type=str, default="sky", choices=["sky", "fluid", "sky+fluid"], help="flownet checkpoint directory")
        self.parser.add_argument("--feature_level", type=int, default=9, help="total number of steps for optimization process")
        
        self.parser.add_argument("--ir_se50", type=str, default="./pretrained_models/model_ir_se50.pth")
        self.parser.add_argument("--moco", type=str, default='./pretrained_models/moco_v2_800ep_pretrain.pt')

        
        ### set inputs
        self.parser.add_argument("--img_path", type=str, help="input image directory")
        self.parser.add_argument("--style_path", type=str, help="style image directory")

        
        ### set save dir
        self.parser.add_argument("--save_dir", type=str, help="image save directory")


        ### cinemagraph
        self.parser.add_argument("--n_frames", type=int, default=120, help="number of frames for looping animation")
        
        
        # stylegan param
        self.parser.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier of the generator. config-f = 2, else = 1")
        
        
        # optim latent param
        self.parser.add_argument("--is_optim",        action="store_true", help="set latent optimization")
        self.parser.add_argument("--optim_step",      type=int, default=3000)
        self.parser.add_argument("--optim_threshold", type=float, default=0.005)
        self.parser.add_argument("--optim_params",    type=str,   default='feat')
        self.parser.add_argument("--initial_lr",      type=float, default=0.1, help="initial learning rate")
        
        
        ### experiment
        self.parser.add_argument('--random_sample', action="store_true", help="Use randomly sampled image or not")
        self.parser.add_argument("--style_interp",  action='store_true', help="Use stylized latent for stylelized cinemagraph generation" )
        self.parser.add_argument("--style_extrapolate_scale", type=float, default=2.0)
        self.parser.add_argument("--mode", type=str, default="full")
        self.parser.add_argument("--recon_feature_idx", type=int, default=9)
        self.parser.add_argument("--warp_feature_idx", type=int, default=9)
        self.parser.add_argument("--vis", type=str, choices=["True", "False"], default="True", help="Visualize intermediate results")
        self.parser.add_argument('--image_inpainting', action='store_true', help="Enable image level inpainting")
        self.parser.add_argument('--no_image_composit', action='store_true', help="Disable Image composition")

    def parse(self):    
        args = self.parser.parse_args()
        return args
