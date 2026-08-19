import os
import sys
import json
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.models as models
from tqdm import tqdm
from datetime import datetime
from Evaluation import evaluation
from models import model_selection
from DatasetLoader import datasetLoader
sys.path.append("../")


global activation
activation = {}

def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output
  return hook

def calculateAlpha(modelLoss, saliencyLoss):
    return modelLoss / (modelLoss + saliencyLoss + 1e-8)


def trainingLoop(attackType):
    """
    
     █████╗ ██████╗  ██████╗     ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ██╗ ██████╗ 
    ██╔══██╗██╔══██╗██╔════╝     ██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗  ██║██╔════╝ 
    ███████║██████╔╝██║  ███╗    ██████╔╝███████║██████╔╝███████╗██║██╔██╗ ██║██║  ███╗
    ██╔══██║██╔══██╗██║   ██║    ██╔═══╝ ██╔══██║██╔══██╗╚════██║██║██║╚██╗██║██║   ██║
    ██║  ██║██║  ██║╚██████╔╝    ██║     ██║  ██║██║  ██║███████║██║██║ ╚████║╚██████╔╝
    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝     ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 

        Train / Test CSVs:
        ------------------
        Artificial_Left_Out_Train.csv        Post_Mortem_Left_Out_Train.csv  Synthetic_Left_Out_Train.csv
        Contacts_+_Print_Left_Out_Train.csv  Printout_Left_Out_Train.csv     Textured_Contact_Left_Out_Train.csv
        Diseased_Left_Out_Train.csv          Glass_Prosthesis_Left_Out_Train.csv 
    """

    # Description of all argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-batchSize', type=int, default=20)
    parser.add_argument('-nEpochs', type=int, default=50)
    parser.add_argument('-csvPath', required=False, default= f'.../CSVs/Train/{attackType}_Left_Out_Train.csv',type=str)
    parser.add_argument('-datasetPath', required=False, default= '',type=str)
    parser.add_argument('-outputPath', required=False, default= '.../VISER_Experiments',type=str)
    parser.add_argument('-heatmaps', required=False, default= '.../Saliency_Files/Eye_Tracking/All_Phases/Blended/',type=str)
    parser.add_argument('-alpha', required=False, default=0.5,type=float)
    parser.add_argument('-network', default= 'densenet',type=str)
    parser.add_argument('-nClasses', default= 2,type=int)

    args = parser.parse_args()
    device = torch.device('cuda')

    print(args)


    """
    
    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗                                                       
    ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║                                                       
    ██╔████╔██║██║   ██║██║  ██║█████╗  ██║                                                       
    ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║                                                       
    ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗                                                  
    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝                                                  
                                                                                                
    █████╗ ██████╗  ██████╗██╗  ██╗██╗████████╗███████╗ ██████╗████████╗██╗   ██╗██████╗ ███████╗
    ██╔══██╗██╔══██╗██╔════╝██║  ██║██║╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗██╔════╝
    ███████║██████╔╝██║     ███████║██║   ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝█████╗  
    ██╔══██║██╔══██╗██║     ██╔══██║██║   ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗██╔══╝  
    ██║  ██║██║  ██║╚██████╗██║  ██║██║   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗
    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝
                                                                                                
    """

    # Definition of model architecture
    im_size = 224
    map_size = 7
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, args.nClasses)
    model = model.to(device)

    print(model)

   # Create destination folder
    os.makedirs(args.outputPath,exist_ok=True)

    # Creation of Log folder: used to save the trained model
    log_path = os.path.join(args.outputPath, 'Logs')
    os.makedirs(log_path, exist_ok=True)


    # Creation of result folder: used to save the performance of trained model on the test set
    result_path = os.path.join(args.outputPath , 'Results')
    os.makedirs(result_path, exist_ok=True)

    class_assgn = {'Live':0,'Spoof':1}

    # Dataloader for train and test data
    dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='Train',c2i=class_assgn,map_location=args.heatmaps,map_size=map_size,im_size=im_size,network=args.network)
    dl = torch.utils.data.DataLoader(dataseta, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
    dataset = datasetLoader(args.csvPath,args.datasetPath, train_test='Test', c2i=dataseta.class_to_id,map_location=args.heatmaps,map_size=map_size,im_size=im_size,network=args.network)
    test = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
    dataloader = {'Train': dl, 'Test':test}


    # Description of hyperparameters
    lr = 0.005
    solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
    lr_sched = optim.lr_scheduler.StepLR(solver, step_size=12, gamma=0.1)

    """
        - Standard classification loss uses Cross Entropy
        - Mean Square Error Loss is used for the image heatmap
        - CYBORG loss combines the two together
    """
    criterion = nn.CrossEntropyLoss()
    criterion_hmap = nn.MSELoss()

    timestamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H_%M")

    # File for logging the training process
    with open(os.path.join(log_path,f'params_{attackType}_Left_Out_{timestamp}.json'), 'w') as Fout:
        hyper = vars(args)
        json.dump(hyper, Fout, indent=4)
    log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}


    """
         ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗ 
         ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝ 
            ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
            ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
            ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
            ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
                                                                
                    ██╗      ██████╗  ██████╗ ██████╗           
                    ██║     ██╔═══██╗██╔═══██╗██╔══██╗          
                    ██║     ██║   ██║██║   ██║██████╔╝          
                    ██║     ██║   ██║██║   ██║██╔═══╝           
                    ███████╗╚██████╔╝╚██████╔╝██║               
                    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝               
    """

    train_loss = []
    test_loss = []
    bestAccuracy = 0
    bestEpoch = 0
    alpha = args.alpha

    if alpha == 1.0:
        print(f"Only using Cross Entropy loss -> Alpha: {alpha}")
    else:
        print(f"Using CYBORG loss -> Alpha: {alpha}")

    train_step = 0
    val_step = 0

    for epoch in range(args.nEpochs):

        for phase in ['Train', 'Test']:
            train = (phase=='Train')
            if phase == 'Train':
                model.train()
                if args.network == "xception":
                    model.model.train()
            else:
                model.eval()
                if args.network == "xception":
                    model.model.eval()
                
            tloss = 0.
            acc = 0.
            tot = 0
            c = 0

            testPredictionScores = []
            testTrueLabel = []
            imageNames = []

            with torch.set_grad_enabled(train):
                for batch_idx, (data, cls, imageName, hmap) in enumerate(tqdm(dataloader[phase])):

                    # Data and ground truth
                    data = data.to(device)
                    cls = cls.to(device)
                    hmap = hmap.to(device)
                    
                    outputs = model(data)

                    # Prediction of accuracy
                    pred = torch.max(outputs,dim=1)[1]
                    corr = torch.sum((pred == cls).int())
                    acc += corr.item()
                    tot += data.size(0)
                    class_loss = criterion(outputs, cls)                       

                    # Running model over data
                    if phase == 'Train' and alpha != 1:
                        if args.network == "densenet":
                            features = model.features(data)
                            params = list(model.classifier.parameters())[0]
                        else:
                            print("INVALID ARCHITECTURE:",args.network)
                            sys.exit()

                        bz, nc, h, w = features.shape

                        beforeDot =  features.reshape((bz, nc, h*w))
                        cams = []
                        for ids,bd in enumerate(beforeDot):
                            weight = params[pred[ids]]
                            cam = torch.matmul(weight, bd)
                            cam_img = cam.reshape(h, w)
                            cam_img = cam_img - torch.min(cam_img)
                            cam_img = cam_img / torch.max(cam_img)
                            cams.append(cam_img)
                        cams = torch.stack(cams)

                        print(f'Min: {hmap.min()} <---> Max: {hmap.max()}')
                        hmap_loss = criterion_hmap(cams,hmap)

                        print(f'Heatmap Loss: {hmap_loss}')

                    else:
                        hmap_loss = 0
                
                    # Optimization of weights for training data
                    if phase == 'Train':
                        if alpha != 1.0:
                            alpha = calculateAlpha(class_loss, hmap_loss)
                            loss = (alpha)*(class_loss) + (1-alpha)*(hmap_loss)
                        else:
                            loss = class_loss
                        train_step += 1
                        solver.zero_grad()

                        loss.backward()
                        solver.step()
                        log['iterations'].append(loss.item())
                      
                    elif phase == 'Test':
                        loss = class_loss
                        val_step += 1
                        temp = outputs.detach().cpu().numpy()
                        scores = np.stack((temp[:,0], np.amax(temp[:,1:args.nClasses], axis=1)), axis=-1)
                        testPredictionScores.extend(scores)
                        testTrueLabel.extend((cls.detach().cpu().numpy()>0)*1)
                        imageNames.extend(imageName)

                    tloss += loss.item()
                    c += 1

            # Logging of train and test results
            if phase == 'Train':
                log['epoch'].append(tloss/c)
                log['train_acc'].append(acc/tot)
                print('Epoch: ', epoch, 'Train loss: ',tloss/c, 'Accuracy: ', acc/tot)
                train_loss.append(tloss / c)

            elif phase == 'Test':
                log['validation'].append(tloss / c)
                log['val_acc'].append(acc / tot)
                print('Epoch: ', epoch, 'Test loss:', tloss / c, 'Accuracy: ', acc / tot)

                lr_sched.step()
                test_loss.append(tloss / c)
                accuracy = acc / tot
              
                if (accuracy >= bestAccuracy):
                    bestAccuracy =accuracy
                    testTrueLabels = testTrueLabel
                    testPredScores = testPredictionScores
                    bestEpoch = epoch
                    save_best_model = os.path.join(log_path,f'final_model_{attackType}_Left_Out_{timestamp}.pth')
                    states = {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': solver.state_dict(),
                    }
                    torch.save(states, save_best_model)
                    testImageNames = imageNames

        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': solver.state_dict(),
        }

        logFile = f'model_log_{attackType}_Left_Out_{timestamp}.json'

        with open(os.path.join(log_path, logFile), 'w') as out:
            json.dump(log, out)
        torch.save(states, os.path.join(log_path,f'current_model_{attackType}_Left_Out_{timestamp}.pth'))


    # Plotting of train and test loss
    plt.figure()
    plt.xlabel('Epoch Count')
    plt.ylabel('Loss')
    plt.plot(np.arange(0, args.nEpochs), train_loss[:], color='r')
    plt.plot(np.arange(0, args.nEpochs), test_loss[:], 'b')
    plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
    plt.savefig(os.path.join(result_path,f'model_Loss_{attackType}_Left_Out_{timestamp}.jpg'))


if __name__ == '__main__':
    normedATs = ["Printout", "Diseased","Post_Mortem","Synthetic",
                 "Contacts_+_Print","Textured_Contact","Artificial"]
    
    for i in range(12):
        for normedAT in normedATs:
            trainingLoop(normedAT)
