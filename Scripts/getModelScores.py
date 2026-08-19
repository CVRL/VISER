import os
import re
import copy
import json
import torch
import argparse
import statistics
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from models import model_selection
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix


class TestDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.image_paths = df['Image_Location'].values

        self.labels = np.array([0 if c.lower() == 'live' else 1 for c in df['Class'].values])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


def contains_pattern(filename, pattern):
    match = re.search(pattern, filename)
    return match is not None
    

def getModelFiles(attackType, modelDirectory, pattern):
    modelFiles = []
    searchPattern = re.escape(f'final_model_{attackType}')

    for modelFile in os.listdir(modelDirectory):
        if contains_pattern(modelFile, searchPattern):
            if contains_pattern(modelFile, re.escape(pattern)):
                continue
            else:
                modelFiles.append(modelFile)

    return modelFiles


def load_densenet121(weights_path, device):
    weights = torch.load(weights_path, map_location=device)
    model = models.densenet121()
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)  # binary classifier
    if "state_dict" in weights:
        model.load_state_dict(weights["state_dict"])
    else:
        model.load_state_dict(weights)
    model.to(device)
    model.eval()
    return model

    
def evaluate_model(model, dataloader, device, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    y_true, y_pred, results = [], [], []

    with torch.no_grad():
        for images, labels, img_paths in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            PAScore = sigmoid(outputs).detach().cpu().numpy()[:, 1]
            SMScore = F.softmax(outputs, dim=1).detach().cpu().numpy()[:, 1]

            if torch.isnan(outputs).any():
                print("NaNs detected directly in model outputs!")
                break

            PAScore = sigmoid(outputs).detach().cpu().numpy()[:, 1]
            SMScore = F.softmax(outputs, dim=1).detach().cpu().numpy()[:, 1]

            for i in range(len(labels)):
                results.append({
                    "Label": int(labels[i].item()),
                    "Image_Path": img_paths[i],
                    "PAScore": float(PAScore[i]),
                    "SoftmaxScore": float(SMScore[i]),
                })

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(PAScore)

    metrics = {}
    
    metrics['AUROC'] = roc_auc_score(y_true, y_pred)
    
    metrics['APCER_1']  = get_apcer_at_fixed_bpcer(y_true, y_pred, target_bpcer=0.01)
    metrics['APCER_5']  = get_apcer_at_fixed_bpcer(y_true, y_pred, target_bpcer=0.05)
    metrics['APCER_10'] = get_apcer_at_fixed_bpcer(y_true, y_pred, target_bpcer=0.10)

    return metrics, results


def get_apcer_at_fixed_bpcer(y_true, y_pred, target_bpcer=0.05):
    """
        Calculates APCER at a specific fixed BPCER threshold.
        y_true: 0 for Live, 1 for Attack
        y_pred: Score (probability of being an attack)
        target_bpcer: The fixed error rate for Live samples (e.g., 0.05 for 5%)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    live_scores = y_pred[y_true == 0]
    attack_scores = y_pred[y_true == 1]

    if len(live_scores) == 0 or len(attack_scores) == 0:
        return 0.0

    threshold = np.percentile(live_scores, (1 - target_bpcer) * 100)

    false_negatives = np.sum(attack_scores < threshold)
    apcer = false_negatives / len(attack_scores)

    return apcer


if __name__ == "__main__":

    backbones = ['Densenet']
    saliencyTypes = ['Baseline', 'Segmentation_Masks', 'ET_All_Phases', 'ET_Initial', 'HA_GB5',
                     'HA_GB10', 'HA_No_Blur', 'HDBSCAN_All_Phases', 'HDBSCAN_Initial']

    normedATs = ["Printout", "Diseased", "Post_Mortem", "Synthetic", 
                 "Contacts_+_Print","Textured_Contact","Artificial"]
    
    jsonDir = f'.../VISER/JSON_Results/Entropy'
    os.makedirs(jsonDir, exist_ok=True)

    jsonResultObject = {
        "totalRuns": 12,
        "backbone": "",
        "configuration": "",
        "attackType": "",
        "AUROC_Scores": [],
        "APCER_at_1_BPCER": [],
        "APCER_at_5_BPCER": [],
        "APCER_at_10_BPCER": []
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for backbone in backbones:
        for saliencyType in saliencyTypes:
            jsonResults = []

            for attackType in normedATs:

                modelDir = f".../VISER/Training_Output/{backbone}/{saliencyType}/Logs"
                testCSV = f".../VISER/CSVs/Test/{attackType}_Left_Out_Test.csv"
                outputDir = f".../VISER/Training_Output/{backbone}/{saliencyType}/Results"

                os.makedirs(outputDir, exist_ok=True)
                outputDir = f".../VISER/Training_Output/{backbone}/{saliencyType}/Results/{attackType}"
                
                batchSize = 20
                os.makedirs(outputDir, exist_ok=True)

                im_size = 224
                
                norm_mean = [0.485, 0.456, 0.406]
                norm_std = [0.229, 0.224, 0.225]

                transform = transforms.Compose([
                    transforms.Resize([im_size, im_size]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=norm_mean, std=norm_std),
                ])

                modelFiles = getModelFiles(attackType, modelDir, pattern='12-05-2026')
                print(f'{len(modelFiles)} model files found')

                auc_list = []
                apcer_1_list = []
                apcer_5_list = []
                apcer_10_list = []

                for modelFile in modelFiles:
                    modelPath = os.path.join(modelDir, modelFile)
                  
                    model = load_densenet121(modelPath, device)

                    dataset = TestDataset(testCSV, transform)
                    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False)

                    metrics, results = evaluate_model(model, dataloader, device)

                    df_results = pd.DataFrame(results)
                    df_results["AUROC"] = metrics['AUROC']

                    auc_list.append(metrics['AUROC'])
                    apcer_1_list.append(metrics['APCER_1'])
                    apcer_5_list.append(metrics['APCER_5'])
                    apcer_10_list.append(metrics['APCER_10'])

                    fileName = modelFile.strip('.pth')
                    fileName = re.sub('final_model_', '', fileName)
                    output_csv = os.path.join(outputDir,fileName)

                    df_results.to_csv(output_csv, index=False)

                print('Results for 12 Runs:')
                print(f'backbone: {backbone}')
                print(f'Saliency Type: {saliencyType}')
                print(f'Attack Type: {attackType}')
                print(f'Avg AUC: {round(statistics.mean(auc_list), 4)}')
                print(f'Avg APCER @ 1% BPCER: {round(statistics.mean(apcer_1_list), 4)}')
                print(f'Avg APCER @ 5% BPCER: {round(statistics.mean(apcer_5_list), 4)}')
                print(f'Avg APCER @ 10% BPCER: {round(statistics.mean(apcer_10_list), 4)}\n')

                jsonResultObject["backbone"] = backbone
                jsonResultObject["configuration"] = saliencyType
                jsonResultObject["attackType"] = attackType
                jsonResultObject["AUROC_Scores"] = auc_list
                jsonResultObject["APCER_at_1_BPCER"] = apcer_1_list
                jsonResultObject["APCER_at_5_BPCER"] = apcer_5_list
                jsonResultObject["APCER_at_10_BPCER"] = apcer_10_list

                jsonResults.append(copy.deepcopy(jsonResultObject))
            
            os.makedirs(f'{jsonDir}/{backbone}', exist_ok=True)
            with open(f'{jsonDir}/{backbone}/{saliencyType}_Results.json', 'w') as Fout:
                json.dump(jsonResults, Fout, indent=4)
