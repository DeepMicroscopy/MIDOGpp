![g206](https://github.com/DeepMicroscopy/MIDOGpp/assets/10051592/934e8017-2c9d-4a2f-9314-885b1aa3a10b)

# MIDOG++: A Comprehensive Multi-Domain Dataset for Mitotic Figure Detection

> The prognostic value of mitotic figures in tumor tissue is well-established for many tumor types and automating this task is of high research interest. 
However, especially deep learning-based methods face performance deterioration in the presence of domain shifts, which may arise from different tumor types, slide preparation and digitization devices. 
We introduce the MIDOG++ dataset, an extension of the MIDOG 2021 and 2022 challenge datasets. We provide region of interest images from 503 histological specimens of seven different tumor types with variable morphology with in total labels for 11,937 mitotic figures: breast carcinoma, lung carcinoma, lymphosarcoma, neuroendocrine tumor, cutaneous mast cell tumor, cutaneous melanoma, and (sub)cutaneous soft tissue sarcoma. The specimens were processed in several laboratories utilizing diverse scanners. 
We evaluated the extent of the domain shift by using state-of-the-art approaches, observing notable differences in single-domain training. In a leave-one-domain-out setting, generalizability improved considerably.
This mitotic figure dataset is the first that incorporates a wide domain shift based on different tumor types, laboratories, whole slide image scanners, and species. 

## Organization of this repository

This repository contains the databases with all mitotic figure annotations of the MIDOG++ dataset alongside the training scripts used in the evaluation. Due to space restrictions, we can't make available the weights of the trained models.

- The [databases/](databases/) folder contains all databases in SQLite [SlideRunner](https://github.com/DeepPathology/SlideRunner) and MS COCO format. 
- The [slide/](slide/) folder contains data loaders for working with whole slide images (WSIs).
- The [images/](images/) folder is empty in the repository, but will be filled by running [Setup.ipynb](Setup.ipynb).

### Getting started

All requirements needed to run the scripts in this repository can be installed using pip:

```pip install -r requirements.txt```

To download all files from figshare, please run the notebook [Setup.ipynb](Setup.ipynb). It will place all 65 GB of images in the images folder.

### Citation

If you use this dataset in your research, please cite our paper:

> Aubreville, M., Wilm, F., Stathonikos, N. et al. A comprehensive multi-domain dataset for mitotic figure detection. Sci Data 10, 484 (2023). https://doi.org/10.1038/s41597-023-02327-4
