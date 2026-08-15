from __future__ import print_function

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

import os
import random
import numpy as np
from tqdm import tqdm
from glob import glob
import SimpleITK as sitk

from options import parse_option
from network import create_model
from utils.transforms import ResizeImage, decode_preds

import warnings
warnings.filterwarnings('ignore')


# Seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

#NOTE: main loop for training
if __name__ == "__main__":

    # Option
    opt = parse_option(print_option=False)

    # Network
    net = create_model(opt)


    # GPU settings
    if opt.use_gpu and torch.cuda.is_available():
        if len(opt.gpu_id) == 1:
            opt.device = torch.device(f'cuda:{opt.gpu_id}')
            net = net.to(opt.device)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
            opt.device = 'cuda'
            net = torch.nn.DataParallel(net).to(opt.device)
    else:
        opt.device = torch.device('cpu')
        net = net.to(opt.device)


    # Load Patient List
    patList = glob(os.path.join(opt.data_root, f'nifti/*{opt.file_type}')) 
    
    # Inference
    for patDir in tqdm(patList):
        patID = patDir.split('/')[-1].split('.')[0]
        dirpath = os.path.join(opt.data_root, 'unet_mask')

        if not os.path.isdir(f'{dirpath}/{patID}'):
            os.makedirs(f'{dirpath}/{patID}', exist_ok=True, mode=0o777)

            # Input Image (FLAIR, T1GD, T1, T2 order)
            imgs = sitk.GetArrayFromImage(sitk.ReadImage(patDir)) # 1, 152, 512, 512

            if opt.vertical_filp:
                imgs = np.swapaxes(np.swapaxes(imgs, 0, 1)[::-1], 0, 1)
            imgs = [imgs]
            
            # Get Shape Information
            org_size = imgs[0].shape # 152, 512, 512
            meta = {'org_size' : torch.Tensor([org_size])}

            # Stack images
            imgs = [img[None, ...] for img in imgs] # 1, 1, 152, 512, 512
            imgs = np.concatenate(imgs, axis=0) 
            imgs = imgs.astype(np.float32)

            # 3D Inference
            if opt.in_dim == 3:
                MEAN = np.array(opt.mean)[:,None,None,None]
                STD = np.array(opt.std)[:,None,None,None]

                imgs = ResizeImage(imgs, (opt.in_depth, opt.in_res, opt.in_res))
                imgs = (imgs - MEAN) / STD

                # Load Data
                imgs = torch.Tensor(imgs[None,...]).float()
                if opt.use_gpu:
                    imgs = imgs.to(opt.device)

                pred = net(imgs).cpu()
            
            # 2D Inference
            elif opt.in_dim == 2:
                
                pred = torch.zeros(opt.n_classes, imgs.shape[1], opt.in_res, opt.in_res) # 3, 152, 512, 512 -> 여기 이상 (152, 3, 512, 512) 아닌가?
                batch_size = opt.batch_size

                for i in range(0, org_size[0], batch_size): # 32, 64, 96, 128

                    # Load Data as Mini-Batch
                    imgs_part = np.moveaxis(imgs[:, i:i + batch_size], 0, 1) # 32, 1, 512, 512  # 여기 이상(i:i+batch_size,:) 아닌가?
                    imgs_part_resized = np.zeros((len(imgs_part), opt.in_channels, opt.in_res, opt.in_res), dtype=np.float32) # (32, 1, 512, 512)

                    for j, single_batch in enumerate(imgs_part): 
                        imgs_part_resized[j] = ResizeImage(single_batch, (opt.in_channels, opt.in_res, opt.in_res)) # 1, 512, 512
                        imgs_part_resized[j] = (imgs_part_resized[j] - imgs_part_resized[j].min()) / (imgs_part_resized[j].max() - imgs_part_resized[j].min())

                    imgs_part_resized = torch.Tensor(imgs_part_resized).float().contiguous() # 32, 1, 512, 512

                    if opt.use_gpu:
                        imgs_part_resized = imgs_part_resized.to(opt.device)

                    with torch.no_grad():
                        pred_part = net(imgs_part_resized).cpu().transpose(0,1) # 32, 3, 512, 512 

                    # Stack mini-batch prediction
                    pred[:, i:i+batch_size] = pred_part # 3, 32, 512, 512
                
                pred = pred[None, ...]

            # Prediction to Original Size and Refine Masks
            pred_decoded = pred.sigmoid().gt(0.5)
            
            # Save Predictions to Disk
            body_pred, muscle_1_pred, muscle_2_pred = pred_decoded.data.numpy().astype(np.uint8)[0]

            if opt.vertical_filp:
                body_pred = np.swapaxes(np.swapaxes(body_pred, 0, 1)[::-1], 0, 1)
                muscle_1_pred = np.swapaxes(np.swapaxes(muscle_1_pred, 0, 1)[::-1], 0, 1)
                muscle_2_pred = np.swapaxes(np.swapaxes(muscle_2_pred, 0, 1)[::-1], 0, 1)

            body_pred, muscle_1_pred, muscle_2_pred = [sitk.GetImageFromArray(array) for array in [body_pred, muscle_1_pred, muscle_2_pred]]

            # Copy Header Information from Original Input Image
            FLAIR_org = sitk.ReadImage(patDir)
            body_pred.CopyInformation(FLAIR_org)
            muscle_1_pred.CopyInformation(FLAIR_org)
            muscle_2_pred.CopyInformation(FLAIR_org)

            # Save Predictions to Disk
            for mask_type, mask_sitk in zip(['body', 'muscle_1', 'muscle_2'], [body_pred, muscle_1_pred, muscle_2_pred]):
                
                mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
                sitk.WriteImage(mask_sitk, f'{dirpath}/{patID}/{mask_type}.nii.gz')
