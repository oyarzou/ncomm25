# RCOR — analysis code and demo dataset

This repository contains Python scripts to compute analysis metrics and reproduce the main figures reported in the manuscript:

**Adaptive recruitment of cortex-wide recurrence for visual object recognition** [BioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.10.17.682937v2.full)

The code operates on derivative files from publicly available BIDS/OpenNeuro datasets and reproduces the quantitative analyses and figures reported in the manuscript. A small standalone demo dataset is included to allow immediate execution of the code without external downloads.

## System requirements
### Software requirements
- Python 3.10
- numpy
- scipy
- matplotlib
- nilearn
- pandas
- seaborn

### Operating systems
Tested on macOS

Expected to run on Linux systems with a standard scientific Python stack

### Hardware
No non-standard hardware required

## Installation guide
### Installation using pip

- `pip install -r requirements.txt`

### Installation using conda

- `conda env create -f environment.yml`
- `conda activate ncomm25`

Typical installation time: approximately 5 minutes on a standard desktop computer.

## Demo
A minimal demo dataset is included in the repository under the directory:

- `demo_data/`

The demo data contain two subjects per modality (EEG and fMRI), organised in BIDS-compatible form and including the derivative files required by the scripts. The demo is intended to verify that the code runs end-to-end and produces representative figures.

To run the demo:

EEG:
- `python eeg.py demo_data/eeg`

fMRI:
- `python fmri.py demo_data/fmri`

Expected output:
- Generation of representative EEG and fMRI figures corresponding to those reported in the manuscript.
- Figures are displayed interactively.

Expected runtime on a standard desktop computer:

- EEG demo: seconds to ~1 minute
- fMRI demo: ~1 minute

## Instructions for use (full data)
To reproduce the full analyses, download the corresponding BIDS datasets from OpenNeuro and point the scripts to the local dataset root.

EEG analysis:
- `python eeg.py <bids_root>`

fMRI analysis:
- `python fmri.py <bids_root>`

where `<bids_root>` is the path to the local BIDS directory containing the required derivative files.

No parameter tuning is required to reproduce the reported results.

## Data availability
The full datasets required to reproduce all quantitative results reported in the manuscript are publicly available via OpenNeuro:
- [EEG Dataset](https://openneuro.org/datasets/ds007162)
- [fMRI Dataset](https://openneuro.org/datasets/ds006805)

Both datasets are provided in BIDS format and include the derivative files used by the scripts in this repository. Accession numbers are reported in the manuscript.

## Notes
- This code is intended for analysis and figure reproduction, not as a general-purpose software package.
- The included demo dataset is illustrative and not intended for statistical inference.
- All analyses reported in the manuscript can be reproduced by running the scripts on the full OpenNeuro datasets.
