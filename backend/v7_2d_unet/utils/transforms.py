from email.mime import image
import cv2
import torch
import random
import numpy as np
import SimpleITK as sitk
from random import uniform
from albumentations import *
from albumentations import DualTransform
from albumentations.augmentations import functional as F
from albumentations.augmentations.geometric.resize import Resize


class UniformRandomResize(DualTransform):
    def __init__(self, scale_range=(0.9, 1.1), interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.interpolation = interpolation

    def get_params_dependent_on_targets(self, params):
        scale = random.uniform(*self.scale_range)
        height = int(round(params['image'].shape[0] * scale))
        width = int(round(params['image'].shape[1] * scale))
        return {'new_height': height, 'new_width': width}

    def apply(self, img, new_height=0, new_width=0, interpolation=cv2.INTER_LINEAR, **params):
        aug = Resize(height=new_height, width=new_width, interpolation=interpolation)
        return aug(image=img)['image']

    def apply_to_keypoint(self, keypoint, new_height=0, new_width=0, **params):
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return "scale_range", "interpolation"

    @property
    def targets_as_params(self):
        return ["image"]


def ResizeSitkImage(sitk_file, new_shape):
    new_shape = (int(new_shape[0]), int(new_shape[1]), int(new_shape[2]))

    new_spacing = [org_spacing*org_size/new_size for org_spacing, org_size, new_size
                   in zip(sitk_file.GetSpacing(), sitk_file.GetSize(), new_shape)]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(sitk_file.GetDirection())
    resample.SetOutputOrigin(sitk_file.GetOrigin())
    resample.SetOutputSpacing(new_spacing)

    resample.SetSize(new_shape)

    return resample.Execute(sitk_file)
    
def ResizeImage(img, new_shape):
    if len(new_shape) == 3:
        if type(img) is np.ndarray:
            new_shape = (new_shape[2], new_shape[1], new_shape[0])
            img = sitk.GetImageFromArray(img)
            img_resized = ResizeSitkImage(img, new_shape)
            return sitk.GetArrayFromImage(img_resized)
        
        else:
            return ResizeSitkImage(img, new_shape)
    
    elif len(new_shape) == 2:
        if np.ndim(img) == 3:
            return cv2.resize(src=img, dsize=new_shape)[None, ...]

        elif np.ndim(img) == 2:
            return cv2.resize(src=img, dsize=new_shape)

def center_crop(img_array, x_size, y_size):
    y, x = img_array.shape[-2:]

    if (y < y_size) or (x < x_size):
        return img_array
        
    x_start = (x//2) - (x_size//2)
    y_start = (y//2) - (y_size//2)
    
    img_crop = img_array[...,
                    y_start : y_start + y_size,
                    x_start : x_start + x_size]

    return img_crop

def augment_imgs_and_masks(img, masks, bright_factor, scale_factor, in_res=512):
    
    # 2D input
    if np.ndim(img) == 3:
        img = img[0]

        augmentator_1 = Compose([
            UniformRandomResize(scale_range=(1-scale_factor, 1+scale_factor)),
            PadIfNeeded(min_height=in_res, min_width=in_res, border_mode=0),
            RandomCrop(in_res, in_res)
        ], p=1.0)
        aug_output = augmentator_1(image=img, mask_bg=masks[0], mask_body=masks[1], mask_muscle_1=masks[2], mask_muscle_2=masks[3])
        img = aug_output['image']

        mask_bg = aug_output['mask_bg']
        mask_body = aug_output['mask_body']
        mask_muscle_1 = aug_output['mask_muscle_1']
        mask_muscle_2 = aug_output['mask_muscle_2']
        masks = np.concatenate([mask_bg[None, ...], mask_body[None, ...], mask_muscle_1[None, ...], mask_muscle_2[None, ...]], axis=0)
        
        augmentator_2 = Compose([
            RandomBrightnessContrast(brightness_limit=(-bright_factor, bright_factor), contrast_limit=(-0.15, 0.4), p=0.75),
        ], p=1.0)
        aug_output = augmentator_2(image=img)

        img = aug_output['image']

    # 3D input
    elif np.ndim(img) == 4:
        pass
    return img[np.newaxis, ...], masks

def random_scale(imgs, masks, max_range):
    scale_factor = uniform(1-max_range, 1+max_range)
    # 2D input
    if np.ndim(imgs) == 3:
        # Move channel axis to last order
        imgs = np.moveaxis(imgs, 0, -1)
        masks = np.moveaxis(masks, 0, -1)

        # Scale images
        imgs_scaled = rescale(imgs, scale_factor, preserve_range=True, mode='edge').astype(imgs.dtype)
        masks_scaled = rescale(masks, scale_factor, preserve_range=True, mode='edge').astype(masks.dtype)

        if scale_factor < 1.0:
            imgs = np.ones_like(imgs, dtype=imgs.dtype) * imgs[0,0]
            masks = np.zeros_like(masks, dtype=masks.dtype)

            imgs[imgs.shape[0]//2-imgs_scaled.shape[0]//2:imgs.shape[0]//2+imgs_scaled.shape[0]//2,
                 imgs.shape[1]//2-imgs_scaled.shape[1]//2:imgs.shape[1]//2+imgs_scaled.shape[1]//2] = imgs_scaled.copy()
            masks[masks.shape[0]//2-masks_scaled.shape[0]//2:masks.shape[0]//2+masks_scaled.shape[0]//2,
                 masks.shape[1]//2-masks_scaled.shape[1]//2:masks.shape[1]//2+masks_scaled.shape[1]//2] = masks_scaled.copy()

        else:
            imgs = imgs_scaled[imgs_scaled.shape[0]//2-imgs.shape[0]//2:imgs_scaled.shape[0]//2+imgs.shape[0]//2,
                               imgs_scaled.shape[1]//2-imgs.shape[1]//2:imgs_scaled.shape[1]//2+imgs.shape[1]//2].copy()
            masks = masks_scaled[masks_scaled.shape[0]//2-masks.shape[0]//2:masks_scaled.shape[0]//2+masks.shape[0]//2,
                               masks_scaled.shape[1]//2-masks.shape[1]//2:masks_scaled.shape[1]//2+masks.shape[1]//2].copy()

        # Recover channel axis to first order
        imgs = np.moveaxis(imgs, -1, 0)
        masks = np.moveaxis(masks, -1, 0)

    # 3D input
    elif np.ndim(imgs) == 4:
        pass

    return imgs, masks

def pad_cropped_boundaries(img_array, x_org, y_org):
    if np.ndim(img_array) == 3:
        re_center_cropped = np.zeros((img_array.shape[0], y_org, x_org))

        z, y, x = img_array.shape

        x_start = (x_org//2) - (x//2)
        y_start = (y_org//2) - (y//2)
        
        re_center_cropped[:,
                y_start : y_start + y,
                x_start : x_start + x] = img_array
    
    elif np.ndim(img_array) == 2:
        re_center_cropped = np.zeros((y_org, x_org))

        y, x = img_array.shape

        x_start = (x_org//2) - (x//2)
        y_start = (y_org//2) - (y//2)
        
        re_center_cropped[
                y_start : y_start + y,
                x_start : x_start + x] = img_array

    return re_center_cropped


def fill_holes(image):
    # Reference : https://stackoverflow.com/questions/50450654/filling-in-circles-in-opencv
    
    n_slices, h, w = image.shape
    image_filled = np.zeros_like(image, dtype=np.uint8)
    
    for d in range(n_slices):
        # Threshold
        th, im_th = cv2.threshold(image[d]*255, 127, 255, cv2.THRESH_BINARY)

        # Copy the thresholded image
        im_floodfill = im_th.copy()

        # Mask used to flood filling.
        # NOTE: the size needs to be 2 pixels bigger on each side than the input image
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground
        im_out = im_th | im_floodfill_inv
        
        image_filled[d] = im_out.copy()
        
    image_filled = (image_filled / 255.).astype(np.uint8)
    return image_filled

def mask_binarization(mask_array, threshold=0):
    # threshold = np.max(mask_array) / 2
    mask_binarized = (mask_array > threshold).astype(np.uint8)
    
    return mask_binarized

def refine_mask(ce_mask, necro_mask, peri_mask):
    total_mask = ((necro_mask != 0) | (ce_mask != 0) | (peri_mask != 0)).astype(np.uint8)
    sum_mask = ((necro_mask != 0) | (ce_mask != 0)).astype(np.uint8)
    
    if np.ndim(total_mask) == 2:
        total_mask = total_mask[None, ...]
        total_mask = fill_holes(total_mask)[0]
        
        sum_mask = sum_mask[None, ...]
        sum_mask = fill_holes(sum_mask)[0]
    else:
        total_mask = fill_holes(total_mask)
        sum_mask = fill_holes(sum_mask)

    refined_peri_array = total_mask.copy()
    refined_peri_array[sum_mask == 1] = 0
    
    refined_ce_mask = sum_mask.copy()
    refined_ce_mask[necro_mask == 1] = 0
    
    return refined_ce_mask, necro_mask, refined_peri_array

def decode_preds(pred, meta=None, refine=False):
    batch_size = pred.size(0)

    if meta is not None:
        org_sizes = meta['org_size'].cpu().data.numpy()

    pred_decoded = []
    for b in range(batch_size):
        # Probability Mask to Binary Mask
        pred_bi = (pred[b].sigmoid() > 0.1).cpu().data.numpy().astype(np.uint8)
        
        # Remove multi-class predicted pixels
        pred_bg, pred_body, pred_muscle_1, pred_muscle_2 = pred_bi
        pred_muscle_2[pred_bg == 1] = 0
        
        if refine:
            pred_body, pred_muscle_1, pred_muscle_2 = refine_mask(pred_body, pred_muscle_1, pred_muscle_2)
        else:
            pred_body[pred_muscle_1 == 1] = 0
            pred_muscle_2[(pred_muscle_1 == 1) | (pred_body == 1)] = 0
        
        # Resize to Original Size
        preds = [pred_body, pred_muscle_1, pred_muscle_2]
        if meta is not None:
            preds = [ResizeImage(pred, org_sizes[b]) for pred in preds]

        preds = [mask_binarization(pred) for pred in preds]
        preds = [pred[None, ...] for pred in preds]

        # Stack processed masks
        pred_bi = np.concatenate(preds, axis=0)
        pred_decoded.append(torch.Tensor(pred_bi).to(pred.device))
    
    return pred_decoded