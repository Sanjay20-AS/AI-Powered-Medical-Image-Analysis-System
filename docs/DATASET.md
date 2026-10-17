# Dataset notes: PneumoniaMNIST

## What it is

PneumoniaMNIST is part of the MedMNIST collection. It is a small educational benchmark derived from pediatric chest X-ray images and distributed in a standardized 28×28 format. This project uses the official train, validation, and test splits provided by the MedMNIST package.

Official project: https://medmnist.com/  
Paper: Yang et al., “MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification,” Scientific Data (2023), https://doi.org/10.1038/s41597-022-01721-8

## How it is downloaded

Run `python scripts/download_dataset.py`. The `medmnist` Python package downloads the dataset. The files are placed beneath `data/medmnist/` and are ignored by Git.

## Important limitations

- Images are resized to 28×28 pixels, much smaller than clinical radiology images.
- This dataset is not a substitute for representative, prospectively collected clinical data.
- Dataset composition, source population, label process, and acquisition characteristics constrain generalization.
- A random split or benchmark test result does not establish external validity.
- Do not publish individual images or derived data without checking applicable terms.
- Do not use this project for diagnosis or patient care.

## Suggested portfolio evidence

Record the dataset version, package version, split sizes, class distribution, and exact command used. Include these in your experiment log. Do not claim a result until you have actually run the experiment.
