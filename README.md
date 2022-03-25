# DIME-Demonstrations
This repository contains the a package which is a part of the official implementation of [DIME](https://arxiv.org/abs/2203.13251) 
for collecting demonstrations for 3 different tasks - rotation, spinning and flipping.

## Setup
This package has the same dependencies as that of the [DIME-Models](https://github.com/NYU-robot-learning/DIME-Models). 
No additional installation is needed.

## Usage
- Run `extractor.py` along with the appropriate parsers to collect a demonstration.
- Run `reengineer.py` along with the appropriate parsers to extract state-action data with specific distance differences.

## Citation
If you use this repo in your research, please consider citing the paper as follows:
```
@article{arunachalam2022dime,
  title={Dexterous Imitation Made Easy: A Learning-Based Framework for Efficient Dexterous Manipulation},
  author={Sridhar Pandian Arunachalam and Sneha Silwal and Ben Evans and Lerrel Pinto},
  journal={arXiv preprint arXiv:2203.13251},
  year={2022}
}
```
