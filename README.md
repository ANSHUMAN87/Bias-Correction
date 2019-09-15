# Bias-Correction
Paper : Data-Free Quantization through Weight Equalization and Bias Correction

Link: https://arxiv.org/pdf/1906.04721v1.pdf

I have implemented only Bias Correction part of the mentioned paper.

I have verified the results on Mobinet_V2 where the results are promising.


# Usage
python3 inference/inference_sim.py -a mobilenet_v2 -b 128 --data "Path-To-Imagenet-Dataset"


# Dataset
Download IMagenet Validation datatset from the official site. Then prepare Imagenet Dataset using the script provided in the script folder.
