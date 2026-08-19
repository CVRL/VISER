# VISER: Visually-Informed System for Enhanced Robustness in Iris Presentation Attack Detection

Official repository for the IEEE Access paper: **[IEEEXplore]() | [ArXiv](https://arxiv.org/abs/2603.17859)**

## Abstract
> Human perceptual priors have shown promise in saliency-guided deep learning training, particularly in the domain of iris presentation attack detection (PAD). Common saliency approaches include hand annotations obtained via mouse clicks and eye gaze heatmaps derived from eye tracking data. However, the most effective form of human saliency for raising generalization to unknown attack classes in iris PAD remains under-explored. In this paper, we conduct a series of experiments comparing hand annotations, eye tracking heatmaps, segmentation masks, and foundation model embeddings to a state-of-the-art deep learning-based baseline on the task of unknown attack type classification for iris PAD. Results in a leave-one-attack-type out paradigm indicate that denoised eye tracking heatmaps show the best generalization improvement over cross entropy in Attack Presentation Classification Error Rate (APCER) at Bona Fide Presentation Classification Error Rate (BPCER) of 1%. Along with this paper, we offer trained models, code, and saliency maps for reproducibility and to facilitate follow-up research efforts.

## Experimental Pipeline
<p align="center">
  <img src="https://github.com/CVRL/VISER/blob/main/Assets/VISER-teaser-v3-1.png?raw=true" />
" width="1000" />
</p>


## Dataset Overview
#### Summary
At a high level, the dataset is organized...

#### Details
The dataset is...

#### Requesting a Copy of the Dataset
Instructions on how to obtain a copy of the dataset can be found at the [Notre Dame's Computer Vision Research Lab webpage](https://cvrl.nd.edu/projects/data/#VISER-2026-dataset) (VISER Dataset). Any questions can be directed to Adam Czajka at aczajka@nd.edu.

## Citation
```
@article{dowling2026viser,
  title={VISER: Visually-Informed System for Enhanced Robustness in Open-Set Iris Presentation Attack Detection},
  author={Dowling, Byron and Piland, Jacob and Frederick, Eleanor and Sweet, Christopher and Czajka, Adam},
  journal={arXiv preprint arXiv:2603.17859},
  year={2026}
}
```

## Acknowledgments

This work was supported by the U.S. Department of Defense (Contract No. W52P1J-20-9-3009) and by the National Science Foundation (Grant No. 2237880). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation, the U.S. Department of Defense or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes, notwithstanding any copyright notation here on.
