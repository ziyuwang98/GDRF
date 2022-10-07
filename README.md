# Generative Deformable Radiance Fields for Disentangled Image Synthesis of Topology-Varying Objects

<p align="center"> 
<img src="/images/teaser.png">
</p>

This repository contains the core implementation of the following paper:

Ziyu Wang, Yu Deng, Jiaolong Yang, Jingyi Yu and Xin Tong

**Generative Deformable Radiance Fields for Disentangled Image Synthesis of Topology-Varying Objects**

Pacific Graphics (PG) & Computer Graphics Forum (CGF), 2022

### [Project page](https://ziyuwang98.github.io/GDRF/) | [Paper](https://arxiv.org/abs/2209.04183) ###

Abstract: _3D-aware generative models have demonstrated their superb performance to generate 3D neural radiance fields (NeRF) from a collection of monocular 2D images even for topology-varying object categories. However, these methods still lack the capability to separately control the shape and appearance of the objects in the generated radiance fields. In this paper, we propose a generative model for synthesizing radiance fields of topology-varying objects with disentangled shape and appearance variations. Our method generates deformable radiance fields, which builds the dense correspondence between the density fields of the objects and encodes their appearances in a shared template field. Our disentanglement is achieved in an unsupervised manner without introducing extra labels to previous 3D-aware GAN training. We also develop an effective image inversion scheme for reconstructing the radiance field of an object in a real monocular image and manipulating its shape and appearance. Experiments show that our method can successfully learn the generative model from unstructured monocular images and well disentangle the shape and appearance for objects (e.g., chairs) with large topological variance. The model trained on synthetic data can faithfully reconstruct the real object in a given single image and achieve high-quality texture and shape editing results._

## Installation
Clone the repository and set up a conda environment with all dependencies as follows:
```
conda env create -f environment.yml
conda activate gdrf
```

## Preparation

We provide the preprocessed training data in [here](https://drive.google.com/drive/folders/1fHOBrWu83zf7HewXjF--wVfHjyBrqV3j?usp=sharing), and the pretrained models in the folder of _pretrained_models_.

## Usage

### Training networks
Run the following script to train a generator from scratch using the preprocessed data:
```
python train.py --curriculum=<CURRICULUM_NAME> --output_dir=<OUTPUT_FOLDER>
```
The code will automatically detect all available GPUs and use DDP training. You can use the default configs provided in the _src/curriculums.py_ or add your own config. To enable training using GPUs with limited memory, you can use patch-level forward and backward process (see [here](https://github.com/microsoft/GRAM/blob/main/images/patch_process.pdf) for a detailed explanation) by changing the _num_regions_ > 1 (e.g. 2, 4, 8, ...) in the curriculums.


### Generating multi-view images with pre-trained models
Run the following script to render multi-view images of generated subjects using a pre-trained model:
```
python render_multiview_images.py --curriculum=<CURRICULUM_NAME> --generator_file=<GENERATOR_PATH.pth> --output_dir=<OUTPUT_FOLDER> --input_seed_list=0,1,2
```
You can add the argument _--input_appear_seed=<SEED_NUMBER>_ to generate images with the same appearance code.
### Evaluation
Run the following script for FID&KID calculation:
```
python fid_evaluation.py --curriculum=<CURRICULUM_NAME> --generator_file=<GENERATOR_PATH.pth> --output_dir=<OUTPUT_FOLDER>
```
By default, 5000 real images and 5000 generated images from EMA model are used for evaluation. 

## License

Licensed under the MIT license.

## Citation

Please cite the following paper if this work helps your research:

    @article{wang2022generative,
        author = {Wang, Ziyu and Deng, Yu and Yang, Jiaolong and Yu, Jingyi and Tong, Xin},
        title = {{Generative Deformable Radiance Fields for Disentangled Image Synthesis of Topology-Varying Objects}},
        journal = {Computer Graphics Forum},
        year = {2022},
    }
## Contact
If you have any questions, please contact Ziyu Wang (wangzy6@shanghaitech.edu.cn).

## Acknowledgements
This implementation takes [pi-GAN](https://github.com/marcoamonteiro/pi-GAN) and [GRAM](https://github.com/microsoft/GRAM) as references. We thank the authors for their excellent work. 
