## File Overview
| Filename    | Description    |
| ---------------| ----------- |
| viser.py | Main driver of the experiment, will perform 12 run per attack type left out for all saliency configurations|
| DatasetLoader.py | Necessary to load and images and paired saliency maps in the main experiment |
| getModelScores.py | Script that will loop over the directory structure set by the training script and calculate AP/BPCER values for trained models|
