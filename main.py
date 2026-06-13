import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from config import Config 
from models import *
from models.clip.clip import load, tokenize
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.datasets import CIFAR100, CIFAR10
from dataset.build_dataset_allsample import build_dataset
from models.clip.simple_tokenizer import SimpleTokenizer

# import matplotlib.pyplot as plt
# from PIL import Image
# from scipy.stats import norm
# from scipy.optimize import curve_fit
# import skimage
# from sklearn.manifold import TSNE

from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
# import matplotlib.colors as colors
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.cm as cm

parser = argparse.ArgumentParser(description='P4Q')
parser.add_argument('--model',
                    choices=[
                        'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64',
                        'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px',
                    ],
                    help='model')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--quant', default=False, action='store_true')
parser.add_argument('--bit_type', 
                    default=8, 
                    type=int,
                    choices=[8, 4, 3, 2])
parser.add_argument('--quant-method',
                    default='minmax',
                    choices=['minmax', 'ema', 'omse', 'percentile'])
parser.add_argument('--calib-batchsize',
                    default=100,
                    type=int,
                    help='batchsize of calibration set')
parser.add_argument('--calib-iter', default=10, type=int)
parser.add_argument('--val-batchsize',
                    default=100,
                    type=int,
                    help='batchsize of validation set')
parser.add_argument('--num-workers',
                    default=4,
                    type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--print-freq',
                    default=50,
                    type=int,
                    help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed')

# base
parser.add_argument("--learn_prompt",default=False, action="store_true")
parser.add_argument("--single_prompt",default=False, action="store_true")
parser.add_argument("--zeroshot_prompt",default=False, action="store_true")

# data
parser.add_argument("--mean_per_class", action='store_true', help='mean_per_class')
parser.add_argument("--db_name", type=str, default='imagenet', help='dataset name') # imagenet
parser.add_argument("--num_runs", type=int, default=1, help='num_runs')
parser.add_argument("--root", type=str, default='./data', help='dataset root')
parser.add_argument("--aug",type=str, default='flip', help='root')
parser.add_argument("--devices", type=int, default=4, help='number of devices')

# optimization setting
parser.add_argument("--coop_lr", type=float, default=5e-4, help='num_runs')
parser.add_argument("--adapter_lr", type=float, default=1e-3, help='num_runs') # 0.001
parser.add_argument("--conv_lr", type=float, default=1e-3, help='num_runs')
parser.add_argument("--wd", type=float, default=0.0, help='num_runs')
parser.add_argument("--coop_epochs", type=int, default=500, help='num_runs')
parser.add_argument("--adapter_epochs", type=int, default=20, help='num_runs')
parser.add_argument("--train_batch", type=int, default=128, help='num_runs')
parser.add_argument("--test_batch", type=int, default=128, help='num_runs')

# learn setting
parser.add_argument("--n_prompt", type=int, default=32, help='num_runs')
parser.add_argument("--prompt_bsz", type=int, default=4, help='num_runs')

# prompt4quant
parser.add_argument("--prompt4quant",default=False, action="store_true")
parser.add_argument("--adapt_ratio", type=float, default=0.2)
parser.add_argument("--val_quant",default=False, action="store_true")

# load & resume
parser.add_argument("--load_dir", type=str, default=None, help='save_path')
parser.add_argument("--resume_checkpoint", type=int, default=0)
parser.add_argument("--save_path", type=str, default=None, help='save_path')
parser.add_argument("--save_freq", type=int, default=0)

# CIFAR 100
CLASSES = [
    'apple',
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'keyboard',
    'lamp',
    'lawn mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak tree',
    'orange',
    'orchid',
    'otter',
    'palm tree',
    'pear',
    'pickup truck',
    'pine tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow tree',
    'wolf',
    'woman',
    'worm',
]

# CIFAR 100 zero-shot templates
templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = tokenize(texts).cuda() #tokenize [80,77]
            class_embeddings = model.encode_text(texts) #embed with text encoder [80, 512]
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True) # [80, 512]
            class_embedding = class_embeddings.mean(dim=0) # [512]
            class_embedding /= class_embedding.norm() # [512]
            zeroshot_weights.append(class_embedding) 
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda() # [n_cls, 512]
    
    return zeroshot_weights

def str2model(name):
    d = {
        'RN50': resnet_50,
        'RN101': resnet_101,
        'RN50x4': resnet_50_4,
        'RN50x16': resnet_50_16,
        'RN50x64': resnet_50_64,
        'ViT-B/32': vit_base_patch32_224,
        'ViT-B/16': vit_base_patch16_224,
        'ViT-L/14': vit_large_patch14_224,
        'ViT-L/14@336px': vit_large_patch14_336,
    }
    print('Model: %s' % d[name].__name__)
    return d[name]

def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = parser.parse_args()
    seed(args.seed)
    # print(args)

    device = torch.device(args.device)
    cfg = Config(args.bit_type, args.quant_method)
    clip_model, preprocess = load(args.model, device=device, cfg=cfg)
    # print(clip_model)
    
    clip_model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
   
    # coop preprocess
    train_db, test_db = build_dataset(args.db_name, args.root, n_shot=1, transform_mode=args.aug)
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=args.train_batch, # 20
        num_workers=args.devices, 
        shuffle=True,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=args.test_batch, # 128
        num_workers=args.devices,
        pin_memory=True,
    )
    
    if args.learn_prompt:
        print("Learning visual and text prompts...")
        data = {'train_loader': train_loader, 'class_names': CLASSES} # CLASSES
        results =[]
        
        for n in range(args.num_runs):
            seed(args.seed+n)
                       # train
            lp_model = ACoOp(clip_model=clip_model, args=args, cfg=cfg)
            lp_model.fit(data)
        
            # evaluation
            acc = lp_model.accuracy(test_loader, mean_per_class=args.mean_per_class) # test with adapter
            results.append({'seed': args.seed+n, 'acc':acc})
            print('Prompt learning accuracy: {}'.format(acc*100))
          
    if args.quant:
        clip_model.model_open_calibrate()
        
        print('Getting caliberation images...') 
        image_list = []
        for i, (data, target) in enumerate(tqdm(train_loader)):
            if i == args.calib_iter:
                break
            data = data.to(device)
            image_list.append(data)
            
        if args.single_prompt:
            print('Single prompt as base..')
            # setting 1: all class caliberate. 
            # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in CLASSES]).to(device) # [100, 77]
            text_inputs = torch.cat([tokenize(f"a photo of the {c}") for c in CLASSES]).to(device)
            with torch.no_grad():
                clip_model.model_open_last_calibrate()
                _ = clip_model.encode_text(text_inputs) # [100, 512]
            # setting 2: some classes caliberate: few-shot
            
            clip_model.model_close_last_calibrate()
           
        elif args.zeroshot_prompt:   
            print('Zeroshot prompts as base...')
            # setting 1: all prompts caliberate.
            # https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
            for i, classname in enumerate(tqdm(CLASSES)):
                texts = [template.format(classname) for template in templates] #format with class
                texts = tokenize(texts).to(device) #tokenize [80,77]
                if i == len(CLASSES) - 1:
                    clip_model.model_open_last_calibrate() # issue: text transformer attention no scale
                class_embeddings = clip_model.encode_text(texts) #embed with text encoder
            
            clip_model.model_close_last_calibrate()
           
        elif args.learn_prompt:
            print('Leanable prompts as base...')
            with torch.no_grad():
                # clip
                lp_model.model.model_open_calibrate()
                
                # text
                print('Caliberating text...') 
                lp_model.model.encode_text.model_open_last_calibrate()
                text_prompt, tokenized_prompts = lp_model.model.text_prompt.to(device), lp_model.model.tokenized_prompts.to(device)
                _ = lp_model.model.encode_text(text_prompt, tokenized_prompts) # calab: text features after quant

                lp_model.model.encode_text.model_close_last_calibrate()

                # image
                lp_model.model.encode_image.model_open_last_calibrate()
                print('Caliberating visual...') 
                for i, image in enumerate(tqdm(image_list)):
                    if i == len(image_list) - 1:
                        lp_model.model.encode_image.model_open_last_calibrate()
                        
                    _ = lp_model.model.encode_image(image)
                lp_model.model.encode_image.model_close_last_calibrate()
                
                # clip
                lp_model.model.model_close_calibrate()
                lp_model.model.model_quant()    
                
        # base not learnt
        if not args.learn_prompt:
            print('Caliberating visual...') 
            with torch.no_grad():
                for i, image in enumerate(tqdm(image_list)):
                    if i == len(image_list) - 1:
                        # This is used for OMSE method to
                        # calculate minimum quantization error
                        clip_model.model_open_last_calibrate()
                    image_features = clip_model.encode_image(image)
            
            clip_model.model_close_calibrate()
         
        # quant model learn: prompt4quant
        if args.prompt4quant:
            data = {'train_loader': train_loader, 'class_names': CLASSES} # CLASSES
            
            if args.single_prompt:
                # single prompt: text_features before quant
                with torch.no_grad():
                    text_features = clip_model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
            elif args.zeroshot_prompt:
                # zero-shot prompt: text_features before quant
                text_features = zeroshot_classifier(clip_model, CLASSES, templates).t() # to check
                
            elif args.learn_prompt:
                # self-learnt prompts: text_features before quant
                text_features = lp_model.model.text_features

            clip_model.model_quant()  # quant
           
            results =[]
            for n in range(args.num_runs):
                seed(args.seed+n)

                if args.learn_prompt:
                    prompt_learner = DQ_QACoOp(qclip=lp_model.model, t_text_features=text_features, args=args)
                else:
                    prompt_learner = DQ_QACoOp(qclip=clip_model, t_text_features=text_features, args=args)
              
                prompt_learner.fit(data) # trainset: all 391 * 128
           
                # evaluation
                acc = prompt_learner.accuracy(test_loader, mean_per_class=args.mean_per_class)
                results.append({'seed': args.seed+n, 'acc':acc})
                print('prompt4quant accuracy: {}'.format(acc*100))
                
                print('Validating P4Q...')
                val_prec1, val_prec5 = validate_p4q(args, test_loader, prompt_learner, device)
                
        else:
            clip_model.model_quant() 
            
    if args.single_prompt:
        text_inputs = torch.cat([tokenize(f"a photo of a {c}") for c in CLASSES]).to(device)
                   
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
    elif args.zeroshot_prompt:
        # zero-shot prompt: : text features after quant
        text_features = zeroshot_classifier(clip_model, CLASSES, templates).t() # to check
        
    elif args.learn_prompt:
        # self-learnt prompts: text features after quant
        with torch.no_grad():
            text_features = lp_model.model.encode_text(text_prompt, tokenized_prompts) # calab: text features after quant
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    if args.val_quant:
        print('Validating quant...')
        if args.learn_prompt:
            val_loss, val_prec1, val_prec5 = validate(args, test_loader, lp_model.model, text_features, criterion, device)
        else:
            val_loss, val_prec1, val_prec5 = validate(args, test_loader, clip_model, text_features, criterion, device)
            
    elif not args.quant:
        val_loss, val_prec1, val_prec5 = validate(args, test_loader, clip_model, text_features, criterion, device)
    
def validate_p4q(args, test_loader, model, device):
    top1 = AverageMeter()
    top5 = AverageMeter()
 
    # switch to evaluate mode
    model.a_lp_qclip.eval()

    for i, (data, target) in enumerate(test_loader):
        
        target = target.to(device)
        data = data.to(device)
        
        with torch.no_grad():
            # pred_y = self.inference(x.cuda())
            pred_y = model.inference(data)
            
        # measure accuracy and record loss
        prec1, prec5 = accuracy(pred_y.data, target, topk=(1, 5))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))


        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i,
                      len(test_loader),
                      top1=top1,
                      top5=top5,
                  ))
            
    val_end_time = time.time()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return  top1.avg, top5.avg

   
def validate(args, test_loader, model, text_features, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
 
    # switch to evaluate mode
    model.eval()
   
    val_start_time = end = time.time()

    for i, (data, target) in enumerate(test_loader):
        
        target = target.to(device)
        data = data.to(device)
        
        with torch.no_grad():
            # predict
            image_features = model.encode_image(data)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            output = logit_scale * image_features @ text_features.t()

            loss = criterion(output, target)
            
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))
        losses.update(loss.data.item(), data.size(0))

            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i,
                      len(test_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                  ))
            
    val_end_time = time.time()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.
          format(top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
  
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().detach()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
   
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
