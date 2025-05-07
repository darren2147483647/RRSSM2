# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from cmath import exp
import tqdm
import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from collections import OrderedDict
import pickle

from compressai.zoo import image_models

import yaml

import numpy as np

import tiny_model
#from tiny_model.predict import tiny_model_RRSSM

## General
'''from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone.fpn import build_resnet_fpn_backbone'''
from utils.dataloader import MSCOCO, Kodak, RIVERAVSSD, RIVERAVSSD_rd

from PIL import Image

## Test
'''from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.data.detection_utils import read_image'''

from contextlib import ExitStack, contextmanager
'''from utils.predictor import ModPredictor'''
from utils.alignment import Alignment

from torch.utils.tensorboard import SummaryWriter

## Function for model to eval
@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
    
    def psnr(self, output, target):
        mse = torch.mean((output - target) ** 2)
        if(mse == 0):
            return 100
        max_pixel = 1.
        psnr = 10 * torch.log10(max_pixel / mse)
        return torch.mean(psnr)

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["rdloss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]
        
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
        return out


class TaskLoss(nn.Module):
    def __init__(self, cfg, device) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        
        self.task_net = tiny_model.tiny_model_RRSSM()#build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3)) #改成RRSSM
        '''checkpoint = OrderedDict()
        with open(cfg.MODEL.WEIGHTS, 'rb') as f:
            FPN_ckpt = pickle.load(f)
            for k, v in FPN_ckpt['model'].items():
                if 'backbone' in k:
                    checkpoint['.'.join(k.split('.')[1:])] = torch.from_numpy(v)
        self.task_net.load_state_dict(checkpoint, strict=True)'''
        self.task_net = self.task_net.to(device)
        for k, p in self.task_net.named_parameters():
            p.requires_grad = False
        self.task_net.eval()
        self.align = Alignment(divisor=32).to(device)
        self.pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(-1, 1, 1).to(device)
        self.pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(-1, 1, 1).to(device)
        # the color space of RRSSM might be RGB(for i modify the pipeline), so I should correct it

    def forward(self, output, d, train_mode=False):
        with torch.no_grad():
            ## Ground truth for perceptual loss
            d = d.mul(255) #no more flip
            d = d - self.pixel_mean
            if not train_mode:
                d = self.align.align(d)
            gt_out = self.task_net(d)
        
        x_hat = torch.clamp(output["x_hat"], 0, 1)
        x_hat = x_hat.mul(255) #no more flip
        x_hat = x_hat - self.pixel_mean
        if not train_mode:
            x_hat = self.align.align(x_hat)
        task_net_out = self.task_net(x_hat)
        
        # print(x_hat.shape,d.shape)
        # print(task_net_out['p2'].shape)
        # exit()
        # from PIL import Image
        # output_dir="."
        # prefix="test20250227_x_hat"
        # for i, img_tensor in enumerate(x_hat):
        #     img_tensor = torch.clamp(img_tensor, 0, 1)  # 限制像素值在 [0, 1] 之間
        #     img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()  # (3, 256, 256) -> (256, 256, 3)
        #     img = Image.fromarray((img_tensor * 255).astype('uint8'))
            
        #     # 儲存圖片
        #     img_path = os.path.join(output_dir, f"{prefix}_{i}.png")
        #     img.save(img_path)
        #     print(f"已儲存圖片至: {img_path}")
        # exit()
        
        # distortion_p2 = nn.MSELoss(reduction='none')(gt_out["p2"], task_net_out["p2"])
        # distortion_p3 = nn.MSELoss(reduction='none')(gt_out["p3"], task_net_out["p3"])
        # distortion_p4 = nn.MSELoss(reduction='none')(gt_out["p4"], task_net_out["p4"])
        # distortion_p5 = nn.MSELoss(reduction='none')(gt_out["p5"], task_net_out["p5"])
        # distortion_p6 = nn.MSELoss(reduction='none')(gt_out["p6"], task_net_out["p6"])
        
        #改extract_feat
        distortion_stage1 = nn.MSELoss()(gt_out[0], task_net_out[0])
        distortion_stage2 = nn.MSELoss()(gt_out[1], task_net_out[1])
        distortion_stage3 = nn.MSELoss()(gt_out[2], task_net_out[2])
        distortion_stage4 = nn.MSELoss()(gt_out[3], task_net_out[3])
        return 0.25*(distortion_stage1+distortion_stage2+distortion_stage3+distortion_stage4)

        return 0.25*(distortion_p2.mean()+distortion_p3.mean()+distortion_p4.mean()+distortion_p5.mean())


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def init(args):
    base_dir = f'{args.root}/{args.exp_name}/{args.quality_level}/'
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def setup_logger(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_file_handler = logging.FileHandler(log_dir, encoding='utf-8')
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    log_stream_handler = logging.StreamHandler(sys.stdout)
    log_stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_stream_handler)

    logging.info('Logging file is %s' % log_dir)


def configure_optimizers(net, args):
    """Set optimizer for only the parameters for propmts"""

    if args.TRANSFER_TYPE == "prompt":
        parameters = {
        k
        for k, p in net.named_parameters()
        if "prompt" in k
    }

    params_dict = dict(net.named_parameters())

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )

    return optimizer

def compute_iou_image(seg_img_tensor, seg_origin_tensor):
    # 確保圖像是二進制格式
    seg_img_tensor = seg_img_tensor.astype(np.bool)
    seg_origin_tensor = seg_origin_tensor.astype(np.bool)
    
    assert seg_img_tensor.shape == seg_origin_tensor.shape, f"{seg_img_tensor.shape}!={seg_origin_tensor.shape}"
    
    # 計算交集（交集區域是兩者都為 1 的區域）
    intersection = np.logical_and(seg_img_tensor, seg_origin_tensor)
    
    # 計算聯集（聯集區域是至少有一個為 1 的區域）
    union = np.logical_or(seg_img_tensor, seg_origin_tensor)
    
    # 計算 IoU
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)
    
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def train_one_epoch(train_dataloader, optimizer, model, criterion_rd, criterion_task, lmbda):
    model.train()
    device = next(model.parameters()).device
    tqdm_emu = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
    for i, d in tqdm_emu:
        d = d.to(device)

        optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion_rd(out_net, d)
        perc_loss = criterion_task(out_net, d)
        total_loss = perc_loss + lmbda * out_criterion['bpp_loss']
        total_loss.backward()
        optimizer.step()

        update_txt=f'[{i*len(d)}/{len(train_dataloader.dataset)}] | Loss: {total_loss.item():.3f} | Distortion loss: {perc_loss.item():.5f} | Bpp loss: {out_criterion["bpp_loss"].item():.4f}'
        tqdm_emu.set_postfix_str(update_txt, refresh=True)


def validation_epoch(epoch, val_dataloader, model, criterion_rd, criterion_task, lmbda):
    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    percloss = AverageMeter()
    totalloss = AverageMeter()

    with torch.no_grad():
        tqdm_meter = tqdm.tqdm(enumerate(val_dataloader),leave=False, total=len(val_dataloader))
        for i, d in tqdm_meter:
            align = Alignment(divisor=256, mode='resize').to(device)

            d = d.to(device)
            align_d = align.align(d)

            out_net = model(align_d)
            out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1)
            out_criterion = criterion_rd(out_net, d)
            perc_loss = criterion_task(out_net, d)
            total_loss = perc_loss + lmbda * out_criterion['bpp_loss']

            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion['psnr'])
            percloss.update(perc_loss)
            totalloss.update(total_loss)

        txt = f"Loss: {totalloss.avg:.3f} | MSE loss: {mse_loss.avg:.5f} | Perception loss: {percloss.avg:.4f} | Bpp loss: {bpp_loss.avg:.4f}"
        tqdm_meter.set_postfix_str(txt)

    model.train()
    print(f"Epoch: {epoch} | bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f}")
    return totalloss.avg


def test_epoch(test_dataloader, model, criterion_rd, predictor, evaluator):
    model.eval()
    device = next(model.parameters()).device
    pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(-1, 1, 1).to(device)

    bpp_loss = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        tqdm_meter = tqdm.tqdm(enumerate(test_dataloader),leave=False, total=len(test_dataloader))
        for i, batch in tqdm_meter:
            with ExitStack() as stack:
                ## model to eval()
                if isinstance(predictor.model, nn.Module):
                    stack.enter_context(inference_context(predictor.model))
                stack.enter_context(torch.no_grad())

                align = Alignment(divisor=256, mode='resize').to(device)
                rcnn_align = Alignment(divisor=32).to(device)

                img = read_image(batch[0]["file_name"], format="BGR")
                d = torch.stack([batch[0]['image'].float().div(255)]).flip(1).to(device)
                align_d = align.align(d)

                out_net = model(align_d)
                out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1)
                out_criterion = criterion_rd(out_net, d)

                trand_y_tilde = out_net['x_hat'].flip(1).mul(255)
                trand_y_tilde = rcnn_align.align(trand_y_tilde - pixel_mean)

                bpp_loss.update(out_criterion["bpp_loss"])
                psnr.update(out_criterion['psnr'])

                ## MaskRCNN
                predictions = predictor(img, trand_y_tilde)
                evaluator.process(batch, [predictions])
            txt = f"Bpp loss: {bpp_loss.avg:.4f} | PSNR loss: {psnr.avg:.4f}"
            tqdm_meter.set_postfix_str(txt)

    results = evaluator.evaluate()
    model.train()
    print(f"bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f}")
    return



import torch
import numpy as np
import matplotlib.pyplot as plt

def show_image_info(x, savename=None):
    # 顯示資料類型與大小
    print(f"Type: {type(x)}, Shape: {x.shape}")

    # 處理 torch tensor 輸入
    if isinstance(x, torch.Tensor):
        if x.dim() == 4:
            x = x[0]  # 取第一張 (C, H, W)
        if x.dim() == 3:
            x = x.permute(1, 2, 0).detach().cpu().numpy()  # 轉為 (H, W, C)

    # 處理 numpy 輸入 (假設已是 HWC)
    if isinstance(x, np.ndarray):
        if x.ndim == 4:
            x = x[0]  # 取第一張 (H, W, C)

    # 顯示圖片
    h, w = x.shape[:2]
    dpi = 100  # 或自己設定成其他數值
    plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)  # 這樣才能 1 pixel 對 1 pixel
    plt.imshow(x.astype(np.uint8) if x.dtype != np.uint8 else x)
    plt.axis('off')
    if savename:
        plt.savefig(f'{savename}.png', bbox_inches='tight', pad_inches=0)
    plt.show()


def inference_epoch(val_dataloader, model, criterion_rd, criterion_task, lmbda, quality_level = None):
    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    percloss = AverageMeter()
    totalloss = AverageMeter()
    
    iou = AverageMeter()
    
    task_net = tiny_model.tiny_model_RRSSM()
    task_net = task_net.to(device)
    for k, p in task_net.named_parameters():
        p.requires_grad = False
    task_net.eval()
    
    batch_size = val_dataloader.batch_size
    
    ckpt_name = "seg_3"
    
    with torch.no_grad():
        tqdm_meter = tqdm.tqdm(enumerate(val_dataloader),leave=False, total=len(val_dataloader))
        for i, d in tqdm_meter:
            align = Alignment(divisor=256, mode='resize').to(device)
            #origin img = 428x240
            d = d.to(device) #torch 8,3,300,535 0~1
            
            check_d = d.cpu()*255
            
            align_d = align.align(d) #torch 8,3,512,768 0~1
            
            check_alignd = align_d.cpu()*255
            
            out_net = model(align_d) #torch 8,3,512,768 0~1
            
            check_rawout = out_net['x_hat'].cpu()*255
            
            out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1) #torch 8,3,300,535 0~1
            
            check_resumeclipout = out_net['x_hat'].cpu()*255
            
            x_hat = out_net['x_hat'].mul(255) #torch 8,3,300,535
            
            check_xhatflipmul = x_hat.cpu()
            
            # x_hat = x_hat - pixel_mean #torch 8,3,300,535
            
            check_xhatnorm = x_hat.cpu()
            
            d255 = d.mul(255) #torch 8,3,300,535
            
            check_flipmuld = d255.cpu()
            
            # d255 = d255 - pixel_mean #torch 8,3,300,535
            
            check_normd = d255.cpu()
            
            gt_out=task_net.inference(d255) #300,535,3
            task_net_out=task_net.inference(x_hat) #300,535,3
            
            check_dfinal0 = gt_out[0].cpu()
            check_xfinal0 = task_net_out[0].cpu()
            
            print([show_image_info(x,f"infer{i}") for i,x in enumerate([check_d,check_alignd,check_rawout,check_resumeclipout,check_xhatflipmul,check_xhatnorm,check_flipmuld,check_normd,check_dfinal0,check_xfinal0])])
            if i==1:
                exit()
            
            from PIL import Image
            output_dir="./inference_img" if quality_level is None else f"./inference_img/{ckpt_name}_{quality_level}"
            prefix="inference"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for ii, (img_tensor,origin_tensor,seg_img_tensor,seg_origin_tensor) in enumerate(zip(out_net['x_hat'],d,task_net_out,gt_out)):
                img_tensor = torch.clamp(img_tensor, 0, 1)  # 限制像素值在 [0, 1] 之間
                img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()  # (3, 256, 256) -> (256, 256, 3)
                img1 = Image.fromarray((img_tensor * 255).astype('uint8'))
                
                origin_tensor = torch.clamp(origin_tensor, 0, 1)  # 限制像素值在 [0, 1] 之間
                origin_tensor = origin_tensor.permute(1, 2, 0).cpu().numpy()  # (3, 256, 256) -> (256, 256, 3)
                img2 = Image.fromarray((origin_tensor * 255).astype('uint8'))
                
                img3 = Image.fromarray(seg_img_tensor.astype('uint8'))
                
                img4 = Image.fromarray(seg_origin_tensor.astype('uint8'))
                
                # 儲存圖片
                img_c = np.concatenate([img2, img1], axis=0)
                img_seg = np.concatenate([img4, img3], axis=0)
                img_overview = np.concatenate([img_c, img_seg], axis=1)
                img_overview = Image.fromarray(img_overview)
                img_path = os.path.join(output_dir, f"{prefix}_{i*batch_size+ii}_result.jpg")
                img_overview.save(img_path)
                
                iou_single_img = compute_iou_image(seg_img_tensor,seg_origin_tensor)
                iou.update(iou_single_img)
                
            out_criterion = criterion_rd(out_net, d)
            perc_loss = criterion_task(out_net, d)
            total_loss = perc_loss + lmbda * out_criterion['bpp_loss']

            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion['psnr'])
            percloss.update(perc_loss)
            totalloss.update(total_loss)

        txt = f"Loss: {totalloss.avg:.3f} | MSE loss: {mse_loss.avg:.5f} | Perception loss: {percloss.avg:.4f} | Bpp loss: {bpp_loss.avg:.4f}"
        tqdm_meter.set_postfix_str(txt)
        

    model.train()
    print(f"INFERENCE | bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f} | iou avg: {iou.avg:.5f}")
    
    if quality_level is not None:
        # 寫入結果檔案
        write_rd_filepath = f"rdresult/{ckpt_name}_{quality_level}.txt"
        with open(write_rd_filepath, 'w') as f:
            f.write(f"{write_rd_filepath},{bpp_loss.avg},{iou.avg}\n")
    
    return totalloss.avg

def inference_epoch_drawrd(val_dataloader, model, criterion_rd, criterion_task, lmbda, quality_level = None, ckpt_name = None):
    print("start inference",torch.cuda.memory_allocated() / 1024 / 1024, "MB")
    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    percloss = AverageMeter()
    totalloss = AverageMeter()
    
    iou = AverageMeter()
    iou2 = AverageMeter()
    
    task_net = tiny_model.tiny_model_RRSSM()
    task_net = task_net.to(device)
    for k, p in task_net.named_parameters():
        p.requires_grad = False
    task_net.eval()
    
    batch_size = val_dataloader.batch_size
    
    ckpt_name = ckpt_name if ckpt_name is not None else "seg"
    print("start2 inference",torch.cuda.memory_allocated() / 1024 / 1024, "MB")
    align = Alignment(divisor=256, mode='resize').to(device)
    with torch.no_grad():
        tqdm_meter = tqdm.tqdm(enumerate(val_dataloader),leave=False, total=len(val_dataloader))
        for i, d in tqdm_meter:
            # print(f"data inference {i}",torch.cuda.memory_allocated() / 1024 / 1024, "MB")
            (d,d2) = d
            
            d = d.to(device)
            #d2 = d2.to(device)
            align_d = align.align(d)
            out_net = model(align_d)
            out_net['x_hat'] = align.resume(out_net['x_hat']).clamp_(0, 1)
            
            x_hat = out_net['x_hat'].mul(255)
            
            d255 = d.mul(255)
            
            gt_out=task_net.inference(d255)
            task_net_out=task_net.inference(x_hat)
            

            output_dir="./inference_img" if quality_level is None else f"./inference_img/{ckpt_name}_{quality_level}"
            prefix="inference"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            #d2 = d2.cpu()
            d2 = [x.permute(1,2,0).numpy() for x in d2]
            #print(task_net_out[0].shape,d2[0].shape)
            for ii, (img_tensor,origin_tensor,seg_img_tensor,seg_origin_tensor,seg_gt_tensor) in enumerate(zip(out_net['x_hat'],d,task_net_out,gt_out,d2)):
                img_tensor = torch.clamp(img_tensor, 0, 1)  # 限制像素值在 [0, 1] 之間
                img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()  # (3, 256, 256) -> (256, 256, 3)
                img1 = Image.fromarray((img_tensor * 255).astype('uint8'))
                
                origin_tensor = torch.clamp(origin_tensor, 0, 1)  # 限制像素值在 [0, 1] 之間
                origin_tensor = origin_tensor.permute(1, 2, 0).cpu().numpy()  # (3, 256, 256) -> (256, 256, 3)
                img2 = Image.fromarray((origin_tensor * 255).astype('uint8'))
                
                img3 = Image.fromarray(seg_img_tensor.astype('uint8'))
                
                img4 = Image.fromarray(seg_origin_tensor.astype('uint8'))
                
                # 儲存圖片
                img_c = np.concatenate([img2, img1], axis=0)
                img_seg = np.concatenate([img4, img3], axis=0)
                img_overview = np.concatenate([img_c, img_seg], axis=1)
                img_overview = Image.fromarray(img_overview)
                img_path = os.path.join(output_dir, f"{prefix}_{i*batch_size+ii}_result.jpg")
                img_overview.save(img_path)
                
                iou_single_img = compute_iou_image(seg_img_tensor,seg_gt_tensor)
                iou.update(iou_single_img)
                
                iou_single_img2 = compute_iou_image(seg_origin_tensor,seg_gt_tensor)
                iou2.update(iou_single_img2)
                
            out_criterion = criterion_rd(out_net, d)
            perc_loss = criterion_task(out_net, d)
            total_loss = perc_loss + lmbda * out_criterion['bpp_loss']

            bpp_loss.update(out_criterion["bpp_loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion['psnr'])
            percloss.update(perc_loss)
            totalloss.update(total_loss)
        del out_net, align_d, x_hat, d255, gt_out, task_net_out
        torch.cuda.empty_cache()

        txt = f"Loss: {totalloss.avg:.3f} | MSE loss: {mse_loss.avg:.5f} | Perception loss: {percloss.avg:.4f} | Bpp loss: {bpp_loss.avg:.4f}"
        tqdm_meter.set_postfix_str(txt)

    print("ending inference",torch.cuda.memory_allocated() / 1024 / 1024, "MB")
    model.train()
    print(f"INFERENCE | bpp loss: {bpp_loss.avg:.5f} | psnr: {psnr.avg:.5f} | iou avg: {iou.avg:.5f} | origin iou avg: {iou2.avg:.5f}")
    
    if quality_level is not None:
        # 寫入結果檔案
        write_rd_filepath = f"rdresult/{ckpt_name}_{quality_level}.txt"
        with open(write_rd_filepath, 'w') as f:
            f.write(f"{write_rd_filepath},{bpp_loss.avg},{iou.avg}\n")
    
    return totalloss.avg

def save_checkpoint(state, is_best, base_dir, filename="checkpoint.pth.tar"):
    logging.info(f"Saving checkpoint: {base_dir+filename}")
    torch.save(state, base_dir+filename)
    if is_best:
        logging.info(f"Saving BEST checkpoint: {base_dir+filename}")
        shutil.copyfile(base_dir+filename, base_dir+"checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-c",
        "--config",
        default="config/vpt_default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        '--name', 
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'), 
        type=str,
        help='Result dir name', 
    )
    given_configs, remaining = parser.parse_known_args(argv)
    with open(given_configs.config) as file:
        yaml_data= yaml.safe_load(file)
        parser.set_defaults(**yaml_data)

    parser.add_argument("-T", "--TEST", action='store_true', help='Testing')
    parser.add_argument("-I", "--INFERENCE", action='store_true', help='Inferencing')
    
    parser.add_argument("--ckptname", type=str, help='for draw rd')
    
    args = parser.parse_args(remaining)
    
    return args


def main(argv):
    args = parse_args(argv)
    base_dir = init(args)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    setup_logger(base_dir + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log')
    msg = f'======================= {args.name} ======================='
    logging.info(msg)
    for k in args.__dict__:
        logging.info(k + ':' + str(args.__dict__[k]))
    logging.info('=' * len(msg))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.dataset=='coco':
        cfg = get_cfg() # get default cfg
        cfg.merge_from_file("./config/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.WEIGHTS = args.maskrcnn_path
    
        det_transformer = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        ## Training
        train_dataset = MSCOCO(args.dataset_path+"/train2017/",
                               det_transformer,
                               "./examples/utils/img_list.txt")
        val_dataset = Kodak(args.dataset_path+"/Kodak/", transforms.ToTensor())

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True,
                                      pin_memory=(device=="cuda"))
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.test_batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    pin_memory=(device=="cuda"))
        
        ## Testing
        if args.TEST:
            json_path = args.dataset_path + "/annotations/instances_val2017.json"
            image_path = args.dataset_path + "/val2017"
            register_coco_instances("compressed_coco", {}, json_path, image_path)
            evaluator = COCOEvaluator("compressed_coco", cfg, False, output_dir="./coco_log")
            evaluator.reset()

            test_dataloader = build_detection_test_loader(cfg, "compressed_coco")
        
            cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNN_with_Rate'
            predictor = ModPredictor(cfg)     
    elif args.dataset=='RIVERAVSSD': #new dataloader
        cfg=None
        
        det_transformer = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        det_transformer_for_small_img = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        det_transformer_better = transforms.Compose([
            transforms.RandomCrop((2048, 2048)),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        det_transformer_small_better = transforms.Compose([
            transforms.Resize((300, 535)),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        det_transformer_small_nocrop = transforms.Compose([
            transforms.Resize((300, 535)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        det_transformer_small_nocrop_noflip = transforms.Compose([
            transforms.Resize((300, 535)),
            transforms.ToTensor()
        ])
        #640x640似乎太大
        
        train_dataset = RIVERAVSSD(args.dataset_path+"/p1img/originalFrames/", det_transformer)
        val_dataset = RIVERAVSSD(args.dataset_path+"/p2img/img/", transforms.ToTensor())
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True,
                                      pin_memory=(device=="cuda"))
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.test_batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    pin_memory=(device=="cuda"))
        if args.TEST:
            test_dataloader = DataLoader(val_dataset,
                                    batch_size=args.test_batch_size,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    pin_memory=(device=="cuda"))
        if args.INFERENCE:
            #inf_dataset = RIVERAVSSD("./dataset/RRSSMtest/", transforms.ToTensor())
            # inf_dataset=RIVERAVSSD(args.dataset_path+"/p2img/img/", det_transformer_small_nocrop_noflip)
            inf_dataset2=RIVERAVSSD_rd(args.dataset_path+"/p2img/img/",args.dataset_path+"/p2label/mask_visual/", transforms.ToTensor())
            # inf_dataloader = DataLoader(inf_dataset,
            #                           batch_size=args.batch_size,
            #                           num_workers=args.num_workers,
            #                           shuffle=False,
            #                           pin_memory=(device=="cuda"))
            inf_dataloader2 = DataLoader(inf_dataset2,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=False,
                                      pin_memory=(device=="cuda"))
            
        # print(f"{len(train_dataloader)} training batch(es) have been loaded")
        # print(f"{len(val_dataloader)} testing batch(es) have been loaded")

    net = image_models[args.model](quality=int(args.quality_level), prompt_config=args)
    net = net.to(device)

    if args.TRANSFER_TYPE == "prompt":
        for k, p in net.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False

    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
    rdcriterion = RateDistortionLoss(lmbda=args.lmbda)
    taskcriterion = TaskLoss(cfg, device)

    last_epoch = 0
    if args.checkpoint: 
        logging.info("Loading "+str(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if list(checkpoint["state_dict"].keys())[0][:7]=='module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                name = k[7:] 
                new_state_dict[name] = v
        else:
            new_state_dict = checkpoint['state_dict']
        net.load_state_dict(new_state_dict, strict=True if args.TEST else False)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    if args.TEST:
        test_epoch(test_dataloader, net, rdcriterion, predictor, evaluator)
        return
    if args.INFERENCE:
        # inference_epoch(inf_dataloader, net, rdcriterion, taskcriterion, args.VPT_lmbda, args.quality_level)
        inference_epoch_drawrd(inf_dataloader2, net, rdcriterion, taskcriterion, args.VPT_lmbda, args.quality_level, args.ckptname)
        return

    import datetime

    log_dir = f"runs/exp1_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    
    best_loss = validation_epoch(-1, val_dataloader, net, rdcriterion, taskcriterion, args.VPT_lmbda)
    tqrange = tqdm.trange(last_epoch, args.epochs)
    for epoch in tqrange:
        train_one_epoch(train_dataloader, optimizer, net, rdcriterion, taskcriterion, args.VPT_lmbda)
        loss = validation_epoch(epoch, val_dataloader, net, rdcriterion, taskcriterion, args.VPT_lmbda)
        lr_scheduler.step()

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        
        writer.add_scalar('Loss/train', loss.item(), epoch+1)
        
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                base_dir,
                filename='checkpoint.pth.tar'
            )
            if epoch%10==9:
                shutil.copyfile(base_dir+'checkpoint.pth.tar', base_dir+ f"checkpoint_{epoch}.pth.tar" )
    writer.close()

if __name__ == "__main__":
    main(sys.argv[1:])
