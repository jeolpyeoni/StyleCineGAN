# StyleCineGAN: Landscape Cinemagraph Generation using a <br> Pre-trained StyleGAN (CVPR 2024)

### [**Paper**](https://arxiv.org/abs/) | [**Project Page**](https://jeolpyeoni.github.io/stylecinegan_project/)

This is the official PyTorch implementation of "StyleCineGAN: Landscape Cinemagraph Generation using a Pre-trained StyleGAN" (CVPR2024).

![teaser](samples/teaser/teaser.gif)
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
The pre-trained model checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1Dkwj5mJOZlkan4U-gdQt6M_JbZXvzFAD?usp=sharing).
<br>Download and unzip the checkpoint files and place them in `./pretrained_models`.

<br>

# Inference
Run main.py to inference the code.

```
python main.py --img_path ./samples/0003835.png --save_dir ./results
```

<br>

## Acknowledgement
The code for this project was build using the codebase of [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch), [pix2pixHD](https://github.com/NVIDIA/pix2pixHD), [FeatureStyleEncoder](https://github.com/InterDigitalInc/FeatureStyleEncoder), [DatasetGAN](https://github.com/nv-tlabs/datasetGAN_release). The `symmetric-splatting` code was built on top of [softmax-splatting](https://github.com/sniklaus/softmax-splatting). We are very thankful to the authors of the corresponding works for releasing their code.

<br>

## Citation
``` bibtex
bibtex code here
```