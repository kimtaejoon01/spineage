import os
import wandb
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from medpy.metric.binary import sensitivity, specificity, dc, hd95

import torch
from torch.autograd import Variable

from utils import AverageMeter
from utils.metrics import DiceCoef
from utils.transforms import decode_preds

def train(net, dataset_trn, optimizer, criterion, epoch, opt):
    print("Start Training...")

    if opt.with_monitor:
        wandb.init(name=opt.exp,
                    project="SNU",
                    entity=opt.user)
        wandb.watch(net, log='all')

    net.train()

    losses = AverageMeter()
    dice_meter_list = []
    for i in range(opt.n_classes):
        dice_meter_list.append(AverageMeter())
    # losses, body_dices, muscle_1_dices, muscle_2_dices, total_dices = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        
    for it, (img, mask) in enumerate(dataset_trn):
        # Optimizer
        optimizer.zero_grad()

        # Load Data
        img, mask = torch.Tensor(img).to(opt.device).float(), torch.Tensor(mask).to(opt.device).float()

        # Predict
        pred = net(img)

        # Loss Calculation
        loss = criterion(pred, mask)

        # Backward and step
        loss.backward()
        optimizer.step()

        # Calculation Dice Coef Score
        pred_decoded = pred.sigmoid().gt(0.5)
        dices = (DiceCoef(return_score_per_channel=True)(pred_decoded, mask))
        total_dice = sum(dices) / opt.n_classes
        # body_dice, muscle_1_dice, muscle_2_dice = DiceCoef(return_score_per_channel=True)(pred_decoded, mask)
        # total_dice = (body_dice + muscle_1_dice + muscle_2_dice) / 3

        for dice, dice_meter in zip(list(dices) + [total_dice], dice_meter_list):
            dice_meter.update(dice.item(), img.size(0))
        # body_dices.update(body_dice.item(), img.size(0))
        # muscle_1_dices.update(muscle_1_dice.item(), img.size(0))
        # muscle_2_dices.update(muscle_2_dice.item(), img.size(0))
        # total_dices.update(total_dice.item(), img.size(0))

        # Stack Results
        losses.update(loss.item(), img.size(0))

        if opt.with_monitor:
            wandb.log({'train_loss': losses.avg, 'train_dice': dice_meter_list[-1].avg})
        print(f'Epoch[{epoch+1}/{opt.max_epoch}] | Iter[{it+1}/{len(dataset_trn)}] | Loss {losses.avg} | Total Dice : {dice_meter_list[-1].avg}')

        # if (it==0) or (it+1) % 10 == 0:
            # print('Epoch[%3d/%3d] | Iter[%3d/%3d] | Loss %.4f | Dice : body %.4f muscle_1 %.4f muscle_2 %.4f Total %.4f'
            #     % (epoch+1, opt.max_epoch, it+1, len(dataset_trn), losses.avg, body_dices.avg, muscle_1_dices.avg, muscle_2_dices.avg, total_dices.avg))

    # print(">>> Epoch[%3d/%3d] | Training Loss : %.4f | Dice : body %.4f muscle_1 %.4f muscle_2 %.4f Total %.4f\n"
    #     % (epoch+1, opt.max_epoch, losses.avg, body_dices.avg, muscle_1_dices.avg, muscle_2_dices.avg, total_dices.avg))
    print(f">>> Epoch{epoch+1}/{opt.max_epoch}] | Training Loss : {losses.avg} | Total Dice : {dice_meter_list[-1].avg}\n")

def validate(dataset_val, net, criterion, optimizer, epoch, opt, best_dice, best_epoch):
    print("Start Evaluation...")
    
    net.eval()

    # 'PatientID - Array' Dictionary
    # body_dict_GT, body_dict_pred = dict(), dict()
    # muscle_1_dict_GT, muscle_1_dict_pred = dict(), dict()
    # muscle_2_dict_GT, muscle_2_dict_pred = dict(), dict()

    # Result containers
    # losses, body_dices, muscle_1_dices, muscle_2_dices, total_dices = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    losses = AverageMeter()
    dice_meter_list = []
    for i in range(opt.n_classes):
        dice_meter_list.append(AverageMeter())
        
    for it, (img, masks_resized, masks_org, meta) in enumerate(dataset_val):
        # Load Data
        if opt.use_gpu:
            img, masks_resized = [tensor.to(opt.device) for tensor in [img, masks_resized]]
        else:
            img, masks_resized = [torch.Tensor(tensor).float() for tensor in [img, masks_resized]]

        # Predict
        with torch.no_grad():
            pred = net(img)

        # Loss Calculation
        loss = criterion(pred, masks_resized)

        # Save GT and Pred Array to Dictionary
        pred_decodeds = pred.sigmoid().gt(0.5) 
        # for pred, gt, patID in zip(pred_decoded, masks_org, meta['patientID']):
        #     pred_body, pred_muscle_1, pred_muscle_2 = pred.cpu().data.numpy()
        #     gt = gt.cpu().data.numpy().reshape(3, 512, 512)
        #     gt_body, gt_muscle_1, gt_muscle_2 = gt[0], gt[1], gt[2] 
        #     patID = int(patID[0].item())
            
        #     if patID not in body_dict_GT:
        #         body_dict_GT[patID] = gt_body[None, ...]
        #         body_dict_pred[patID] = pred_body[None, ...]
        #         muscle_1_dict_GT[patID] = gt_muscle_1[None, ...]
        #         muscle_1_dict_pred[patID] = pred_muscle_1[None, ...]
        #         muscle_2_dict_GT[patID] = gt_muscle_2[None, ...]
        #         muscle_2_dict_pred[patID] = pred_muscle_2[None, ...]
                
        #     else:
        #         body_dict_GT[patID] = np.concatenate([body_dict_GT[patID], gt_body[None, ...]], 0)
        #         body_dict_pred[patID] = np.concatenate([body_dict_pred[patID], pred_body[None, ...]], 0)
        #         muscle_1_dict_GT[patID] = np.concatenate([muscle_1_dict_GT[patID], gt_muscle_1[None, ...]], 0)
        #         muscle_1_dict_pred[patID] = np.concatenate([muscle_1_dict_pred[patID], pred_muscle_1[None, ...]], 0)
        #         muscle_2_dict_GT[patID] = np.concatenate([muscle_2_dict_GT[patID], gt_muscle_2[None, ...]], 0)
        #         muscle_2_dict_pred[patID] = np.concatenate([muscle_2_dict_pred[patID], pred_muscle_2[None, ...]], 0)

        # Stack Results
        losses.update(loss.item(), img.size(0))
    
    # Evaluation Metrics Calculation
    for pred_decoded, gt in zip(pred_decodeds, masks_resized):
        dice_list = []
        pred_decoded = pred_decoded.cpu().data.numpy()
        gt = gt.cpu().data.numpy()
        
        for i, (pred_decoded_one_cls, gt_one_cls) in enumerate(zip(pred_decoded, gt)):
            dice = dc(pred_decoded_one_cls, gt_one_cls)
            dice_list.append(dice)
            
        for dice, dice_meter in zip(dice_list, dice_meter_list[:-1]):
            dice_meter.update(dice, 1)
        dice_meter_list[-1].update(sum(dice_list) / opt.n_classes, 1)
        
    # for patID in body_dict_GT:
    #     gt_body = body_dict_GT[patID]
    #     pred_body = body_dict_pred[patID]
        
    #     gt_muscle_1 = muscle_1_dict_GT[patID]
    #     pred_muscle_1 = muscle_1_dict_pred[patID]
        
    #     gt_muscle_2 = muscle_2_dict_GT[patID]
    #     pred_muscle_2 = muscle_2_dict_pred[patID]

    #     body_dice = dc(pred_body, gt_body)
    #     muscle_1_dice = dc(pred_muscle_1, gt_muscle_1)
    #     muscle_2_dice = dc(pred_muscle_2, gt_muscle_2)
    #     total_dice = (body_dice + muscle_1_dice + muscle_2_dice) / 3

    #     body_dices.update(body_dice, 1)
    #     muscle_1_dices.update(muscle_1_dice, 1)
    #     muscle_2_dices.update(muscle_2_dice, 1)
    #     total_dices.update(total_dice, 1)

    if opt.with_monitor:
        wandb.log({'valid_loss': losses.avg, 'valid_dice': dice_meter_list[-1].avg})

    print(f">>> Epoch[{epoch+1}/{opt.max_epoch}] | Test Loss : {losses.avg} | Total Dice : {dice_meter_list[-1].avg}")
    # print(">>> Epoch[%3d/%3d] | Test Loss : %.4f | Dice : body %.4f muscle_1 %.4f muscle_2 %.4f Total %.4f"
    #     % (epoch+1, opt.max_epoch, losses.avg, body_dices.avg, muscle_1_dices.avg, muscle_2_dices.avg, dice_meter_list[-1].avg))

    # Update Result
    if dice_meter_list[-1].avg > best_dice:
        print('Best Score Updated...')
        best_dice = dice_meter_list[-1].avg
        best_epoch = epoch

        # Remove previous weights pth files
        for path in glob('exp/%s_*best_dice*.pth' % opt.exp):
            os.remove(path)

        model_filename = 'exp/%s_epoch_%04d_best_dice%.4f_loss%.8f.pth' % (opt.exp, epoch+1, best_dice, losses.avg)

        # Single GPU
        if opt.ngpu == 1:
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)

    print('>>> Current best: Dice: %.8f in %3d epoch\n' % (best_dice, best_epoch+1))
    
    if (epoch+1) % int(opt.save_epoch) == 0:
        model_filename = 'exp/%s_epoch_%04d_dice%.4f_loss%.8f.pth' % (opt.exp, epoch+1, dice_meter_list[-1].avg, losses.avg)
        if opt.ngpu == 1:
            torch.save(net.state_dict(), model_filename)
        # Multi GPU
        else:
            torch.save(net.module.state_dict(), model_filename)
    
    return best_dice, best_epoch


def evaluate(dataset_val, net, opt):
    print("Start Evaluation...")

    net.eval()

    # 'PatientID - Array' Dictionary
    body_dict_GT, body_dict_pred = dict(), dict()
    muscle_1_dict_GT, muscle_1_dict_pred = dict(), dict()
    muscle_2_dict_GT, muscle_2_dict_pred = dict(), dict()

    # Result containers
    body_dices, muscle_1_dices, muscle_2_dices = AverageMeter(), AverageMeter(), AverageMeter()
    body_hausdorff95s, muscle_1_hausdorff95s, muscle_2_hausdorff95s = AverageMeter(), AverageMeter(), AverageMeter()
    body_sensitivity, muscle_1_sensitivity, muscle_2_sensitivity = AverageMeter(), AverageMeter(), AverageMeter()
    body_specificity, muscle_1_specificity, muscle_2_specificity = AverageMeter(), AverageMeter(), AverageMeter()
    
    iter = 0
    for img, _, masks_org, meta in tqdm(dataset_val):
        # Load Data
        img = torch.Tensor(img).float()

        if opt.use_gpu:
            img = img.to(opt.device)

        # Predict
        with torch.no_grad():
            pred = net(img)

        # Save GT and Pred Array to Dictionary
        pred_decoded = pred.sigmoid().gt(0.5)

        for pred, gt, patID in zip(pred_decoded, masks_org, meta['patientID']):
            pred_body, pred_muscle_1, pred_muscle_2 = pred.cpu().data.numpy()
            gt = gt.cpu().data.numpy().reshape(3, 512, 512)
            gt_body, gt_muscle_1, gt_muscle_2 = gt[0], gt[1], gt[2]
            patID = int(patID[0].item())
            
            if patID not in body_dict_GT:
                body_dict_GT[patID] = gt_body[None, ...]
                body_dict_pred[patID] = pred_body[None, ...]
                muscle_1_dict_GT[patID] = gt_muscle_1[None, ...]
                muscle_1_dict_pred[patID] = pred_muscle_1[None, ...]
                muscle_2_dict_GT[patID] = gt_muscle_2[None, ...]
                muscle_2_dict_pred[patID] = pred_muscle_2[None, ...]
                
            else:
                body_dict_GT[patID] = np.concatenate([body_dict_GT[patID], gt_body[None, ...]], 0)
                body_dict_pred[patID] = np.concatenate([body_dict_pred[patID], pred_body[None, ...]], 0)
                muscle_1_dict_GT[patID] = np.concatenate([muscle_1_dict_GT[patID], gt_muscle_1[None, ...]], 0)
                muscle_1_dict_pred[patID] = np.concatenate([muscle_1_dict_pred[patID], pred_muscle_1[None, ...]], 0)
                muscle_2_dict_GT[patID] = np.concatenate([muscle_2_dict_GT[patID], gt_muscle_2[None, ...]], 0)
                muscle_2_dict_pred[patID] = np.concatenate([muscle_2_dict_pred[patID], pred_muscle_2[None, ...]], 0)
        iter += 1

    # Calculate Metrics
    print('\nCalculating Evaluation Metrics...')
    for patID, gt_body in body_dict_GT.items():
        pred_body = body_dict_pred[patID]
        body_dices.update(dc(pred_body, gt_body), 1)
        body_hausdorff95s.update(hd95(gt_body, pred_body), 1)
        body_sensitivity.update(sensitivity(gt_body, pred_body), 1)
        body_specificity.update(specificity(gt_body, pred_body), 1)

    for patID, gt_muscle_1 in muscle_1_dict_GT.items():
        pred_muscle_1 = muscle_1_dict_pred[patID]
        muscle_1_dices.update(dc(pred_muscle_1, gt_muscle_1), 1)
        muscle_1_hausdorff95s.update(hd95(gt_muscle_1, pred_muscle_1), 1)
        muscle_1_sensitivity.update(sensitivity(gt_muscle_1, pred_muscle_1), 1)
        muscle_1_specificity.update(specificity(gt_muscle_1, pred_muscle_1), 1)
        
    for patID, gt_muscle_2 in muscle_2_dict_GT.items():
        pred_muscle_2 = muscle_2_dict_pred[patID]
        muscle_2_dices.update(dc(pred_muscle_2, gt_muscle_2), 1)
        muscle_2_hausdorff95s.update(hd95(gt_muscle_2, pred_muscle_2), 1)
        muscle_2_sensitivity.update(sensitivity(gt_muscle_2, pred_muscle_2), 1)
        muscle_2_specificity.update(specificity(gt_muscle_2, pred_muscle_2), 1)

    print("Evaluate Result\
           \n>>>> Dice : body %.4f muscle_1 %.4f muscle_2 %.4f\
           \n>>>> Hausdorff95 : body %.4f muscle_1 %.4f muscle_2 %.4f\
           \n>>>> Sensitivity : body %.4f muscle_1 %.4f muscle_2 %.4f\
           \n>>>> Specificity : body %.4f muscle_1 %.4f muscle_2 %.4f\
           "
        % (body_dices.avg, muscle_1_dices.avg, muscle_2_dices.avg,
           body_hausdorff95s.avg, muscle_1_hausdorff95s.avg, muscle_2_hausdorff95s.avg,
           body_sensitivity.avg, muscle_1_sensitivity.avg, muscle_2_sensitivity.avg,
           body_specificity.avg, muscle_1_specificity.avg, muscle_2_specificity.avg,))