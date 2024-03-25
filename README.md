# StyleCineGAN: Landscape Cinemagraph Generation using a Pre-trained StyleGAN (CVPR 2024)

### [**Paper**](https://arxiv.org/abs/2403.14186) | [**Project Page**](https://jeolpyeoni.github.io/stylecinegan_project/)

This is the official PyTorch implementation of "StyleCineGAN: Landscape Cinemagraph Generation using a Pre-trained StyleGAN" (CVPR2024).

![teaser](teaser/teaser.gif)
> Abstract: We propose a method that can generate cinemagraphs automatically from a still landscape image using a pre-trained StyleGAN. Inspired by the success of recent unconditional video generation, we leverage a powerful pre-trained image generator to synthesize high-quality cinemagraphs. Unlike previous approaches that mainly utilize the latent space of a pre-trained StyleGAN, our approach utilizes its deep feature space for both GAN inversion and cinemagraph generation. Specifically, we propose multi-scale deep feature warping (MSDFW), which warps the intermediate features of a pre-trained StyleGAN at different resolutions. By using MSDFW, the generated cinemagraphs are of high resolution and exhibit plausible looping animation. We demonstrate the superiority of our method through user studies and quantitative comparisons with state-of-the-art cinemagraph generation methods and a video generation method that uses a pre-trained StyleGAN.

<br>

# Getting Started


## Environment Setup
We recommend to use Docker. Use **seokg1023/vml-pytorch:vessl** for the docker image. 
```bash
docker pull seokg1023/vml-pytorch:vessl
```
<br>

All dependencies for the environment are provided in requirements.txt.
```bash
pip install -r requirements.txt
```

## Download Checkpoints
We provide pre-trained checkpoints of StyleGAN2 and encoder networks [here](https://drive.google.com/drive/folders/1Dkwj5mJOZlkan4U-gdQt6M_JbZXvzFAD?usp=sharing).
<br>Download and unzip the checkpoint files and place them in `./pretrained_models`.

<br>

# Inference
We provide an inference code for the proposed MSDFW method following the GAN inversion process.
Run main.py as the following example:

```
python main.py --img_path ./samples/0002268 --save_dir ./results
```

To test the method with your own data, please place the data as below:
```
$IMG_PATH$
    └── $FILE_NAME$
         ├── $FILE_NAME$.png
         ├── $FILE_NAME$_mask.png
         └── $FILE_NAME$_motion.npy
```
   


<br>

## Acknowledgement
The code for this project was build using the codebase of [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch), [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), [FeatureStyleEncoder](https://github.com/InterDigitalInc/FeatureStyleEncoder), [DatasetGAN](https://github.com/nv-tlabs/datasetGAN_release). The `symmetric-splatting` code was built on top of [softmax-splatting](https://github.com/sniklaus/softmax-splatting). We are very thankful to the authors of the corresponding works for releasing their code.

<br>

## Citation
``` bibtex
@misc{choi2024stylecinegan,
        title={StyleCineGAN: Landscape Cinemagraph Generation using a Pre-trained StyleGAN}, 
        author={Jongwoo Choi and Kwanggyoon Seo and Amirsaman Ashtari and Junyong Noh},
        year={2024},
        eprint={2403.14186},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
  }
```
