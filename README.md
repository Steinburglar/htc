<div align="center">
<a href="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/htc_logo.svg"><img src="https://e130-hyperspectal-tissue-classification.s3.dkfz.de/figures/htc_logo.png" alt="Logo" width="800" /></a>

![Python](https://img.shields.io/badge/python-3.9%20|%203.10-brightgreen)
</div>

# Hyperspectral Tissue Classification
This package is a framework for automated tissue classification and segmentation on medical hyperspectral imaging (HSI) data. It contains:

- The implementation of deep learning models to solve supervised classification and segmentation problems for a variety of different input spatial granularities (pixels, superpixels, patches and entire images) and modalities (RGB data, raw and processed HSI data) from our paper [<q>Robust deep learning-based semantic organ segmentation in hyperspectral images</q>](https://doi.org/10.1016/j.media.2022.102488).
- Corresponding pretrained models.
- A pipeline to efficiently load and process HSI data, to aggregate deep learning results and to validate and visualize findings.

This framework is designed to work on HSI data from the [Tivita](https://diaspective-vision.com/en/) cameras but you can adapt it to different HSI datasets as well. Potential applications include:

<!-- TODO: link to public dataset for all occurences of HeiPorSPECTRAL -->
- Use our data loading and processing pipeline to easily access image and meta data for any work utilizing Tivita datasets.
 - This repository is tightly coupled to work with the soon-to-be public HeiPorSPECTRAL dataset. If you already downloaded the data, you only need to perform the setup steps and then you can directly use the `htc` framework to work on the data (cf. [our tutorials](#tutorials)).
- Train your own networks and benefit from a pipeline offering e.g. efficient data loading, correct hierarchical aggregation of results and a set of helpful visualizations.
- Apply deep learning models for different spatial granularities and modalities on your own semantically annotated dataset.
- Use our pretrained models to initialize the weights for your own training.
- Use our pretrained models to generate predictions for your own data.

If you use the `htc` framework, please cite our paper [<q>Robust deep learning-based semantic organ segmentation in hyperspectral images</q>](https://doi.org/10.1016/j.media.2022.102488):

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@article{SEIDLITZ2022102488,
    title = {Robust deep learning-based semantic organ segmentation in hyperspectral images},
    journal = {Medical Image Analysis},
    volume = {80},
    pages = {102488},
    year = {2022},
    issn = {1361-8415},
    doi = {10.1016/j.media.2022.102488},
    url = {https://www.sciencedirect.com/science/article/pii/S1361841522001359},
    author = {Silvia Seidlitz and Jan Sellner and Jan Odenthal and Berkin Özdemir and Alexander Studier-Fischer and Samuel Knödler and Leonardo Ayala and Tim J. Adler and Hannes G. Kenngott and Minu Tizabi and Martin Wagner and Felix Nickel and Beat P. Müller-Stich and Lena Maier-Hein}
}
```
</details>

## Setup
### Package Installation
This package can be installed via pip:
```bash
pip install imsy-htc
```
This installs all the required dependencies defined in [`requirements.txt`](requirements.txt). The requirements include [PyTorch](https://pytorch.org/), so you may want to install it manually before installing the package in case you have specific needs (e.g. CUDA version).

> &#x26a0;&#xfe0f; This framework was developed and tested using the Ubuntu 20.04+ Linux distribution. Despite we do provide wheels for Windows and macOS as well, they are not tested.

> &#x26a0;&#xfe0f; Network training and inference was conducted using a RTX 3090 GPU with 24 GiB of memory. It should also work with GPUs which have less memory but you may have to adjust some settings (e.g. the batch size).

<details close>
<summary>Optional Dependencies (<code>imsy-htc[extra]</code>)</summary>

Some requirements are considered optional (e.g. if they are only needed by certain scripts) and you will get an error message if they are needed but unavailable. You can install them via
```bash
pip install --extra-index-url https://read_package:CnzBrgDfKMWS4cxf-r31@git.dkfz.de/api/v4/projects/15/packages/pypi/simple imsy-htc[extra]
```
or by adding the following lines to your `requirements.txt`
```
--extra-index-url https://read_package:CnzBrgDfKMWS4cxf-r31@git.dkfz.de/api/v4/projects/15/packages/pypi/simple
imsy-htc[extra]
```

This installs the optional dependencies defined in [`requirements-extra.txt`](requirements-extra.txt), including for example our Python wrapper for the [challengeR toolkit](https://github.com/wiesenfa/challengeR).
</details>

<details close>
<summary>Docker</summary>

We also provide a Docker setup for testing. As a prerequisite:
- Clone this repository
- Install [Docker](https://docs.docker.com/get-docker/) and [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- Install the required dependencies to run the Docker startup script:
```bash
pip install python-dotenv
```

Make sure that your environment variables are available and then bash into the container
```bash
export PATH_Tivita_HeiPorSPECTRAL="/path/to/the/dataset"
python run_docker.py bash
```
You can now run any commands you like. All datasets you provided via an environment variable that starts with `PATH_Tivita` will be accessible in your container (you can also check the generated `docker-compose.override.yml` file for details). Please note that the Docker container is for small testing only and not for development. This is also reflected by the fact that per default all results are stored inside the container and hence will also be deleted after exiting the container. If you want to keep your results, let the environment variable `PATH_HTC_DOCKER_RESULTS` point to the directory where you want to store the results.
</details>

<details close>
<summary>Developer Installation</summary>

If you want to make changes to the package code (which is highly welcome 😉), we recommend to install the `htc` package in editable mode in a separate conda environment:

```bash
# Set up the conda environment
# Note: you might want to consider adding conda-forge to your default channels (if not done already)
#   conda config --add channels conda-forge
# so that you also get the latest patch releases for Python
conda create --yes --name htc python=3.10
conda activate htc

# Install the htc package and its requirements
pip install -r requirements-dev.txt
pip install --no-use-pep517 -e .
```

Before commiting any files, please run the static code checks locally
```bash
git add .
pre-commit run --all-files
```
</details>

### Environment Variables
This framework can be configured via environment variables. Most importantly, we need to know where your data is located (e.g. `PATH_Tivita_HeiPorSPECTRAL`) and where results should be stored (e.g. `PATH_HTC_RESULTS`). For a full list of possible environment variables, please have a look at the documentation of the [`Settings`](htc/settings.py) class.

> 💡 If you set an environment variable for a dataset path, it is important that the variable name matches the folder name (e.g. the variable name `PATH_Tivita_HeiPorSPECTRAL` matches the dataset path `my/path/HeiPorSPECTRAL` with its folder name `HeiPorSPECTRAL`, whereas the variable name `PATH_Tivita_some_other_name` does not match). Furthermore, the dataset path needs to point to a directory which contains a `data` and an `intermediates` subfolder.

There are several options to set the environment variables. For example:
- You can specify a variable as part of your bash startup script `~/.bashrc` or before running each command:
    ```bash
    PATH_HTC_RESULTS="~/htc/results" htc training --model image --config "models/image/configs/default"
    ```
    However, this might get cumbersome or does not give you the flexibility you need. 
- Recommended if you cloned this repository (in contrast to simply installing it via pip): You can create a `.env` file in the repository root and fill it with your variables, for example:
    ```bash
    export PATH_Tivita_HeiPorSPECTRAL=/mnt/nvme_4tb/HeiPorSPECTRAL
    export PATH_HTC_RESULTS=~/htc/results
    ```
- Recommended if you installed the package via pip: You can create user settings for this application. The location is OS-specific. For Linux the location might be at `~/.config/htc/variables.env`. Please run `htc info` upon package installation to retrieve the exact location on your system. The content of the file is the same as the .env above.

After setting your environment variables, it is recommended to run `htc info` to check that your variables are correctly registered in the framework.

## Tutorials
A series of [tutorials](tutorials) can help you get started on the `htc` framework by guiding you through different usage scenarios.
> 💡 The tutorials make use of our public HSI dataset HeiPorSPECTRAL. If you want to directly run them, please download the dataset first and make it accessible via the environment variable `PATH_Tivita_HeiPorSPECTRAL` as described above.

- As a start, we recommend to take a look at this [general notebook](tutorials/General.ipynb) which showcases the basic functionalities of the `htc` framework. Namely, it demonstrates the usage of the `DataPath` class which is the entry point to load and process HSI data. For example, you will learn how to read HSI cubes, segmentation masks and meta data. Among others, you can use this information to calculate the median spectrum of an organ.
- If you want to use our framework with your own dataset, it might be necessary to write a custom `DataPath` class so that you can find your images and annotations. We [collected some tips](tutorials/CustomDataPath.md) on how this can be achieved.
- You have some HSI data at hand and want to use one of our pretrained models to generate predictions? Then our [prediction notebook](tutorials/CreatingPredictions.ipynb) has got you covered.
- You want to use our pretrained models to initialize the weights for your own training? You can use the [PyTorch Hub](https://pytorch.org/hub/) for this. See the section about [pretrained models](#pretrained-models) below for details.
- You want to use our framework to train a network? The [network training notebook](tutorials/network_training/NetworkTraining.ipynb) will teach you everything you need using the example of a heart and lung segmentation network.
- There is also a [notebook with low-level details](tutorials/FileReference.ipynb) on our public HSI dataset. This is useful if you want to know more about the underlying data structure. Or maybe you stumbled upon a file and want to know how to read it.
- If you are interested in our technical validation (e.g. because you want to compare your colorchecker images with ours) and need to create a mask to detect the different colorchecker fields, you might find our automatic [colorchecker mask creation pipeline](htc/utils/ColorcheckerMaskCreation.ipynb) useful.

We do not have a separate documentation website for our framework yet. However, most of the functions and classes are documented so feel free to explore the source code or use your favorite IDE to display the documentation. If something does not become clear from the documentation, feel free to open an issue!

## Pretrained Models
This framework gives you access to a variety of pretrained segmentation and classification models. The models will be automatically downloaded, provided you specify the model type (e.g. `image`) and the run folder (e.g. `2022-02-03_22-58-44_generated_default_model_comparison`). It can then be used for example to [create predictions](tutorials/CreatingPredictions.ipynb) on some data or as a baseline for your own training (see example below).

The following table lists all the models you can get:
| model type | modality | run folder | class |
| ----------- | ----------- | ----------- | ----------- |
| image | hsi | [2022-02-03_22-58-44_generated_default_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_model_comparison.zip) | [`ModelImage`](htc/models/image/ModelImage.py) |
| image | param | [2022-02-03_22-58-44_generated_default_parameters_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip) | [`ModelImage`](htc/models/image/ModelImage.py) |
| image | rgb | [2022-02-03_22-58-44_generated_default_rgb_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/image@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip) | [`ModelImage`](htc/models/image/ModelImage.py) |
| patch | hsi | [2022-02-03_22-58-44_generated_default_64_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_model_comparison.zip) | [`ModelImage`](htc/models/image/ModelImage.py) |
| patch | param | [2022-02-03_22-58-44_generated_default_64_parameters_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_parameters_model_comparison.zip) | [`ModelImage`](htc/models/image/ModelImage.py) |
| patch | rgb | [2022-02-03_22-58-44_generated_default_64_rgb_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_64_rgb_model_comparison.zip) | [`ModelImage`](htc/models/image/ModelImage.py) |
| patch | hsi | [2022-02-03_22-58-44_generated_default_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_model_comparison.zip) | [`ModelImage`](htc/models/image/ModelImage.py) |
| patch | param | [2022-02-03_22-58-44_generated_default_parameters_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip) | [`ModelImage`](htc/models/image/ModelImage.py) |
| patch | rgb | [2022-02-03_22-58-44_generated_default_rgb_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/patch@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip) | [`ModelImage`](htc/models/image/ModelImage.py) |
| superpixel_classification | hsi | [2022-02-03_22-58-44_generated_default_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_model_comparison.zip) | [`ModelSuperpixelClassification`](htc/models/superpixel_classification/ModelSuperpixelClassification.py) |
| superpixel_classification | param | [2022-02-03_22-58-44_generated_default_parameters_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip) | [`ModelSuperpixelClassification`](htc/models/superpixel_classification/ModelSuperpixelClassification.py) |
| superpixel_classification | rgb | [2022-02-03_22-58-44_generated_default_rgb_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/superpixel_classification@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip) | [`ModelSuperpixelClassification`](htc/models/superpixel_classification/ModelSuperpixelClassification.py) |
| pixel | hsi | [2022-02-03_22-58-44_generated_default_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_model_comparison.zip) | [`ModelPixel`](htc/models/pixel/ModelPixel.py) |
| pixel | param | [2022-02-03_22-58-44_generated_default_parameters_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_parameters_model_comparison.zip) | [`ModelPixelRGB`](htc/models/pixel/ModelPixelRGB.py) |
| pixel | rgb | [2022-02-03_22-58-44_generated_default_rgb_model_comparison](https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/pixel@2022-02-03_22-58-44_generated_default_rgb_model_comparison.zip) | [`ModelPixelRGB`](htc/models/pixel/ModelPixelRGB.py) |

> 💡 The modality `param` refers to stacked tissue parameter images (named TPI in our paper [<q>Robust deep learning-based semantic organ segmentation in hyperspectral images</q>](https://doi.org/10.1016/j.media.2022.102488)). For the model type `patch`, pretrained models are available for the patch sizes 64 x 64 and 32 x 32 pixels. The modality and patch size is not specified when loading a model as it is already characterized by specifying a certain run folder.

You can use [`torch.hub.list`](https://pytorch.org/docs/stable/hub.html#torch.hub.list) and [`torch.hub.help`](https://pytorch.org/docs/stable/hub.html#torch.hub.help) to get more information on how to use the models to initialize the weights for your own training. As a teaser, consider the following example which loads the pretrained image HSI network:
```python
import torch

model = torch.hub.load("IMSY-DKFZ/htc", "image", run_folder="2022-02-03_22-58-44_generated_default_model_comparison", n_channels=100, n_classes=19)
input_data = torch.randn(1, 100, 480, 640)  # NCHW
model(input_data).shape
# torch.Size([1, 19, 480, 640])
```

> 💡 Please note that when initializing the weights as described above, the segmentation head is initialized randomly. Meaningful predictions on your own data can thus not be expected out of the box, but you will have to train the model on your data first.

## CLI
There is a common command line interface for many scripts in this repository. More precisely, every script which is prefixed with `run_NAME.py` can also be run via `htc NAME` from any directory. For more details, just type `htc`.

## Papers
This repository contains code to reproduce our publications listed below:

### 📝 Robust deep learning-based semantic organ segmentation in hyperspectral images
[https://doi.org/10.1016/j.media.2022.102488](https://doi.org/10.1016/j.media.2022.102488)

In this paper, we trained several segmentation networks and compared different spatial granularities (e.g. patch vs. image) and modalities (e.g. HSI vs. RGB). Furthermore, we give insights into the required amount of training dta or the generalization capabilities of the models across subjects. The pretrained networks are related to this paper. You can find the notebooks to generate the paper figures in [paper/MIA2021](paper/MIA2021) (the folder also includes a [reproducibility document](paper/MIA2021/reproducibility.md)) and the models in [htc/models](htc/models). For each model, there are three configuration files, namely default, default_rgb and default_parameters, which correspond to the HSI, RGB and TPI modality, respectively.

> 📂 The dataset for this paper is not publicly available.

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@article{SEIDLITZ2022102488,
    title = {Robust deep learning-based semantic organ segmentation in hyperspectral images},
    journal = {Medical Image Analysis},
    volume = {80},
    pages = {102488},
    year = {2022},
    issn = {1361-8415},
    doi = {10.1016/j.media.2022.102488},
    url = {https://www.sciencedirect.com/science/article/pii/S1361841522001359},
    author = {Silvia Seidlitz and Jan Sellner and Jan Odenthal and Berkin Özdemir and Alexander Studier-Fischer and Samuel Knödler and Leonardo Ayala and Tim J. Adler and Hannes G. Kenngott and Minu Tizabi and Martin Wagner and Felix Nickel and Beat P. Müller-Stich and Lena Maier-Hein}
}
```
</details>

### 📝 Spectral organ fingerprints for machine learning-based intraoperative tissue classification with hyperspectral imaging in a porcine model
[https://doi.org/10.1038/s41598-022-15040-w](https://doi.org/10.1038/s41598-022-15040-w)

In this paper, we trained a classification model based on median spectra. You can find the model code in [htc/tissue_atlas](htc/tissue_atlas) and the confusion matrix figure of the paper in [paper/NatureReports2021](paper/NatureReports2021).

> 📂 The dataset for this paper is not fully publicly available. Part of the data used in this paper overlaps with the open HeiPorSPECTRAL dataset.

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@article{Studier-Fischer2022,
    author = {Studier-Fischer, Alexander and Seidlitz, Silvia and Sellner, Jan and Özdemir, Berkin and Wiesenfarth, Manuel and Ayala, Leonardo and Odenthal, Jan and Knödler, Samuel and Kowalewski, Karl Friedrich and Haney, Caelan Max and Camplisson, Isabella and Dietrich, Maximilian and Schmidt, Karsten and Salg, Gabriel Alexander and Kenngott, Hannes Götz and Adler, Tim Julian and Schreck, Nicholas and Kopp-Schneider, Annette and Maier-Hein, Klaus and Maier-Hein, Lena and Müller-Stich, Beat Peter and Nickel, Felix},
    title = {Spectral organ fingerprints for machine learning-based intraoperative tissue classification with hyperspectral imaging in a porcine model},
    journal = {Scientific Reports},
    year = {2022},
    month = {Jun},
    day = {30},
    volume = {12},
    number = {1},
    pages = {11028},
    issn = {2045-2322},
    doi = {10.1038/s41598-022-15040-w},
    url = {https://doi.org/10.1038/s41598-022-15040-w}
}
```
</details>

<!-- TODO: adjust title once set, change soon-to-be -->
### 📝 HeiPorSPECTRAL - A dataset for hyperspectral imaging data of 20 physiological organs in a porcine model

This paper introduces the HeiPorSPECTRAL dataset containing 5756 hyperspectral images from 11 subjects. We are using these images as examples for our tutorials. You can find the visualization notebook for the paper figures in [paper/NatureData2023](paper/NatureData2023) (the folder also includes a [reproducibility document](paper/NatureData2023/reproducibility.md)) and the remaining code in [htc/tissue_atlas_open](htc/tissue_atlas_open).

> 📂 The dataset for this paper is soon-to-be publicly available.

### 📝 Künstliche Intelligenz und hyperspektrale Bildgebung zur bildgestützten Assistenz in der minimal-invasiven Chirurgie
[https://doi.org/10.1007/s00104-022-01677-w](https://doi.org/10.1007/s00104-022-01677-w)

You can find the code generating our figure for this paper at [paper/Chirurg2022](paper/Chirurg2022).

> 📂 The data used in this paper is the same as for <q>Robust deep learning-based semantic organ segmentation in hyperspectral images</q> and hence not publicly available.

<details closed>
<summary>Cite via BibTeX</summary>

```bibtex
@article{Chalopin2022,
    author = {Chalopin, Claire and Nickel, Felix and Pfahl, Annekatrin and Köhler, Hannes and Maktabi, Marianne and Thieme, René and Sucher, Robert and Jansen-Winkeln, Boris and Studier-Fischer, Alexander and Seidlitz, Silvia and Maier-Hein, Lena and Neumuth, Thomas and Melzer, Andreas and Müller-Stich, Beat Peter and Gockel, Ines},
    title = {Künstliche Intelligenz und hyperspektrale Bildgebung zur bildgestützten Assistenz in der minimal-invasiven Chirurgie},
    journal = {Die Chirurgie},
    year = {2022},
    month = {Oct},
    day = {01},
    volume = {93},
    number = {10},
    pages = {940-947},
    issn = {2731-698X},
    doi = {10.1007/s00104-022-01677-w},
    url = {https://doi.org/10.1007/s00104-022-01677-w}
}
```
</details>