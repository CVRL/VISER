# VISER: Visually-Informed System for Enhanced Robustness in Iris Presentation Attack Detection

Official repository for the paper: Byron Dowling, Jacob Piland, Eleanor Frederick, Christopher Sweet and Adam Czajka, "VISER: Visually-Informed System for Enhanced Robustness in Iris Presentation Attack Detection," IEEE/IAPR International Joint Conference on Biometrics, Rome, Italy, September 1-4, 2026 **([ArXiv](https://arxiv.org/abs/2603.17859) | [IEEEXplore]())**


## Abstract
> Human perceptual priors have shown promise in saliency-guided deep learning training, particularly in the domain of iris presentation attack detection (PAD). Common saliency approaches include hand annotations obtained via mouse clicks and eye gaze heatmaps derived from eye tracking data. However, the most effective form of human saliency for raising generalization to unknown attack classes in iris PAD remains under-explored. In this paper, we conduct a series of experiments comparing hand annotations, eye tracking heatmaps, segmentation masks, and foundation model embeddings to a state-of-the-art deep learning-based baseline on the task of unknown attack type classification for iris PAD. Results in a leave-one-attack-type out paradigm indicate that denoised eye tracking heatmaps show the best generalization improvement over cross entropy in Attack Presentation Classification Error Rate (APCER) at Bona Fide Presentation Classification Error Rate (BPCER) of 1%.

## Experimental Pipeline
<p align="center">
  <img src="https://github.com/CVRL/VISER/blob/main/Assets/VISER-teaser-v3-1.png?raw=true" width="1000" />
</p>


## Dataset Overview
#### Summary
At a high level, the dataset is a JSON file where each object includes:
* A reference to the original iris image
* Ground truth label
* Attack type category
* Links to saliency maps for each tested configuration for the image

#### Details
The dataset is organized as a list of JSON objects where each object refers to an iris sample from the dataset described in the paper. Each JSON object contains a reference to the iris sample and the attack type the sample represents with Live indicating Bonafide, or Spoof indicating some type of presentation attack category. Additionally there is a dictionary of image links that correspond to the different saliency map configurations that are associated with this image, i.e. *"Denoised_Initial_ET": "Denoised_Initial_ET/9_5_blended.png"* is a reference to the de-noised initial eye tracking saliency map for the image 9_5.png that was tested during the experiments.

#### Example JSON Object
```json
    {
        "irisImageLink": "9_5.png",
        "label": "Live",
        "attackType": "Live",
        "saliencyMaps": {
            "Denoised_Initial_ET": "Denoised_Initial_ET/9_5_blended.png",
            "Denoised_Full_ET": "Denoised_Full_ET/9_5_blended.png",
            "Initial_Eye_Tracking": "Initial_Eye_Tracking/9_5_blended.png",
            "Full_Eye_Tracking": "Full_Eye_Tracking/9_5_blended.png",
            "Segmentation_Masks": "Segmentation_Masks/9_5.png",
            "Hand_Annotations_Low_Entropy": "Hand_Annotations_Low_Entropy/9_5_blended.png",
            "Hand_Annotations_Equal_Entropy": "Hand_Annotations_Equal_Entropy/9_5_blended.png",
            "Hand_Annotations_High_Entropy": "Hand_Annotations_High_Entropy/9_5_blended.png"
        }
    }
```

#### Requesting a Copy of the Dataset
Researchers interested in obtaining a copy of the data associated with the paper are requested to execute the [data sharing license agreement](Assets/ND-IJCB26-VISER-license.pdf). **Note for university licensees:** We cannot accept licenses signed by students or postdoctoral scholars under any circumstances. We cannot accept licenses signed by faculty members unless they have been explicitly delegated the authority to make contracts on behalf of the institution. Your institution's legal or contracting office must review and execute the license. 

## Scripts and Model Weights

Scripts to replicate the experiments presented in the paper are in the [Scripts](Scripts/) folder. Required model weights can be downloaded from [this Google Drive folder](https://drive.google.com/drive/folders/1kUhVMcFDGftVShtkUUIT33vs6JVQnk9F?usp=sharing).

## Citation
```
@inproceedings{dowling2026viser,
  title={VISER: Visually-Informed System for Enhanced Robustness in Open-Set Iris Presentation Attack Detection},
  author={Dowling, Byron and Piland, Jacob and Frederick, Eleanor and Sweet, Christopher and Czajka, Adam},
  year={2026},
  booktitle={IEEE/IAPR International Joint Conference on Biometrics, Rome, Italy, September 1-4, 2026},
}
```

## Acknowledgments

This work was supported by the U.S. Department of Defense (Contract No. W52P1J-20-9-3009) and by the National Science Foundation (Grant No. 2237880). Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation, the U.S. Department of Defense or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes, notwithstanding any copyright notation here on.
