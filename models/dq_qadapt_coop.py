import torch
import torch.nn as nn
from torch.nn import functional as F
import os

from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pdb
from models.clip.clip import load, tokenize
from models.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

from .utils import build_cosine_scheduler
from .ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear, QMultiheadAttention

from collections import OrderedDict
from typing import Tuple, Union
from functools import partial

def save_params(model, path):
    params = {
        'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(params, path)
    
def load_params(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '') if key.startswith('module.') else key
        cleaned[new_key] = value
    model.load_state_dict(cleaned, strict=False)

def resolve_checkpoint_paths(load_dir, resume_checkpoint):
    p4q_pl = os.path.join(load_dir, 'p4q_prompt_learner.pth')
    p4q_ad = os.path.join(load_dir, 'p4q_adapter.pth')
    if os.path.isfile(p4q_pl) and os.path.isfile(p4q_ad):
        return p4q_pl, p4q_ad
    return (
        os.path.join(load_dir, f'epoch_{resume_checkpoint}_prompt_learner.pth'),
        os.path.join(load_dir, f'epoch_{resume_checkpoint}_adapter.pth'),
    )

def load_val_params(model, path):
    checkpoint = torch.load(path)
    # import pdb
    # pdb.set_trace()
    state_dict = checkpoint["model_state_dict"]

    # Ignore fixed token vectors, only the context embeddings remain
    if "module.token_prefix" in state_dict:
        del state_dict["module.token_prefix"]

    if "module.token_suffix" in state_dict:
        del state_dict["module.token_suffix"]
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
   
class PromptLearner(nn.Module):
    def __init__(self, class_names,ctx_dim, clip_model, n_ctx=16, prompt_pos=2, device=None):
        super().__init__()
        dtype = clip_model.dtype

        n_cls = len(class_names)
        self.dtype = dtype
        
        ctx_vectors = torch.empty(1, n_ctx, ctx_dim, dtype=self.dtype) # [1, n_ctx, ctx_dim]
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors).to(device)

        prompt_prefix =' '.join(['x'] * n_ctx) # 'x x x x x x x x x x x x x x x x'
        prompts = [prompt_prefix + ' ' + name + '.' for name in class_names] # 'x x x x x x x x x x x x x x x x toilet paper.', before training

        classnames = [name.replace('_', ' ') for name in class_names]
        self.name_lens = [len(_tokenizer.encode(name)) for name in class_names]

        self.prompt_pos = prompt_pos

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(device) # [1000, 77]
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype) # [1000, 77, 512]
        self.register_buffer( 'token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim], [1000, 1, 512], embeddings of the SOS (start of sequence) tokens for each class;  correspond to the first token in the tokenized prompts
        self.register_buffer( 'token_suffix', embedding[:, 1+n_ctx:,:]) # CLS, EOS, [n_cls, -1, ctx_dim], [1000, 60, 512]; represents the embeddings of the tokens that come after the n_ctx context tokens in each prompt

        self.n_cls = n_cls 
        self.n_ctx = n_ctx 
        self.ctx_dim = ctx_dim

    def forward(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param.device}")
            
        ctx=self.ctx
        tokenized_prompts = self.tokenized_prompts.view(self.n_cls,-1) # [1000, 77]
        n_cls = self.n_cls
        
        if self.prompt_pos == 2:
            prefix = self.token_prefix.unsqueeze(1) # [1000, 1, 1, 512]
            suffix = self.token_suffix.unsqueeze(1) # [1000, 1, 60, 512]
            ctx = ctx.unsqueeze(0).repeat(n_cls, 1, 1, 1) # [1000, 1, 16, 512], replace ctx in token_embedding
            prompts = torch.cat([prefix, ctx, suffix],dim=2) # [1000, 77, 512]
        elif self.prompt_pos == 1:
            prompts =[]
            half_n_ctx = self.n_ctx // 2
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1, :,:].unsqueeze(1)
                class_i = self.token_suffix[i:i+1,:name_len, :].unsqueeze(1)
                suffix_i = self.token_suffix[i:i+1, name_len:,:].unsqueeze(1)
                ctx_i_half1 = ctx[:,:half_n_ctx, :].unsqueeze(0)
                ctx_i_half2 = ctx[:, half_n_ctx:,:].unsqueeze(0)
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.prompt_pos == 0:
            prompts =[]
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1,:,:].unsqueeze(1)
                class_i = self.token_suffix[i:i+1, :name_len,:].unsqueeze(1)
                suffix_i = self.token_suffix[i:i+1, name_len:,:].unsqueeze(1)
                ctx_i = ctx.unsqueeze(0)
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        prompts = prompts.view(n_cls, -1, self.ctx_dim)

        return prompts, tokenized_prompts # prompt_embeddings (part replaced), tokenized_prompts

class QAdapter(nn.Module):
    def __init__(self, c_in, reduction=4, cfg=None):
        super(QAdapter, self).__init__()
        self.cfg = cfg
        self.fc = nn.Sequential(OrderedDict([
            ('act_in_fc', QAct( 
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A_attn,
            calibration_mode=cfg.CALIBRATION_MODE_A_v, # attention activation
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
            )), # input quant
            ("in_fc", QLinear(c_in, c_in // reduction, quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_W_attn,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W
            )),
            ("relu", nn.ReLU(inplace=True)),
            ('act_out_fc', QAct(
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A_attn,
            calibration_mode=cfg.CALIBRATION_MODE_A_v, #  attention activation
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
            )), # input quant
            ("out_fc", QLinear(c_in // reduction, c_in, quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_W_attn,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W))
            ]))
                                
    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.quant = True
                
    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.last_calibrate = True
    
    def model_close_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.last_calibrate = False
                        
    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.calibrate = False
             
    def forward(self, x):
        x = self.fc(x)
        return x
    
class LSQ_Adapter(nn.Module):
    def __init__(self, c_in, reduction=4, cfg=None):
        super(LSQ_Adapter, self).__init__()
        self.cfg = cfg
        self.fc = nn.Sequential(OrderedDict([
            ('act_in_fc', ActLSQ( 
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A_attn,
            calibration_mode=cfg.CALIBRATION_MODE_A_v, # attention activation
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
            )), # input quant
            ("in_fc", LinearLSQ(c_in, c_in // reduction, quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_W_attn,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W
            )),
            ("relu", nn.ReLU(inplace=True)),
            ('act_out_fc', ActLSQ(
            quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_A_attn,
            calibration_mode=cfg.CALIBRATION_MODE_A_v, #  attention activation
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
            )), # input quant
            ("out_fc", LinearLSQ(c_in // reduction, c_in, quant=False,
            calibrate=False,
            bit_type=cfg.BIT_TYPE_W_attn,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W))
            ]))
                                
    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.quant = True
                
    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.last_calibrate = True
    
    def model_close_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.last_calibrate = False
                        
    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.calibrate = False
             
    def forward(self, x):
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, args, clip_model, device=None):
        super().__init__()
        if args.learn_prompt:
            self.transformer = clip_model.encode_text.transformer # learn_prompt
            self.positional_embedding = clip_model.encode_text.positional_embedding
            self.ln_final = clip_model.encode_text.ln_final
            self.text_projection = clip_model.encode_text.text_projection
        else:
            self.transformer = clip_model.transformer  # bert original 
            self.positional_embedding = clip_model.positional_embedding
            self.ln_final = clip_model.ln_final
            self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype # torch.float32
        self.cfg = clip_model.cfg
        self.device = device
        
    def model_quant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.last_calibrate = True
    
    def model_close_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.last_calibrate = False
                        
    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
                m.calibrate = False
               
    def forward(self, x, tokenized_prompts):
        # x.to(self.device)
        x = x + self.positional_embedding.type(self.dtype) #position_embeding可训练
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection # @ and
        return x

class CLIP(nn.Module):
    def __init__(self, args, class_names, clip_model, n_ctx=16, cfg=None, device=None):
        super().__init__()
        self.n_class = len(class_names)
        self.encode_text = TextEncoder(args, clip_model, device=device) # reset into self.encode_text
           
        # dp
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), using all...')
            self.encode_text = nn.DataParallel(self.encode_text).to(device)
         
        if args.learn_prompt:
            self.ctx_dim = clip_model.encode_text.ln_final.weight.shape[0] # text
            self.encode_image = clip_model.encode_image.visual # image
        else:
            self.ctx_dim = clip_model.ln_final.weight.shape[0]  # text
            self.encode_image = clip_model.visual # image
            
        self.prompt_learner = PromptLearner(class_names=class_names, 
                        ctx_dim=self.ctx_dim,
                        clip_model=clip_model, 
                        n_ctx=n_ctx, 
                        device=device)
        if device_count > 1:
           self.prompt_learner =  nn.DataParallel(self.prompt_learner).to(device)
          

        self.logit_scale = clip_model.logit_scale
        self.cur_text_prompt = None

        # adapter
        self.adapt_ratio = args.adapt_ratio
        self.adapter = QAdapter(self.ctx_dim, 4, cfg).to(device)

    def forward(self, image, test=False):
        if test:
            # image
            with torch.no_grad():
                # test with adapter
                image_features = self.encode_image(image.type(self.dtype)) # image features after quant
                x = self.adapter(image_features) 
                image_features = self.adapt_ratio * x + (1 - self.adapt_ratio) * image_features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.detach()
            
            # text
            text_features = self.text_features
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits

        else:
            # train adapter
            # for m in self.encode_image.modules():
            #     if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax, QMultiheadAttention]:
            #         print('Module name: {}, status:{}'.format(type(m), m.quant))
                    # Module name: <class 'models.ptq.layers.QAct'>, status:True
                    # Module name: <class 'models.ptq.layers.QLinear'>, status:True
                  
            image_features = self.encode_image(image.type(self.dtype)) # image features after quant
            x = self.adapter(image_features)
            image_features = self.adapt_ratio * x + (1 - self.adapt_ratio) * image_features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # image_features = image_features.detach()
            
            # train PromptLearner
            text_prompt, tokenized_prompts = self.prompt_learner() # forward
            text_features = self.encode_text(text_prompt,tokenized_prompts) # text features after quant
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(self.n_class, -1)
           
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            
            # distill
            self.cur_text_prompt = text_prompt

            return logits

    @torch.no_grad()
    def set_classifier(self):
        text_prompt, tokenized_prompts = self.prompt_learner() # forward
        try:
            text_features = self.encode_text(text_prompt, tokenized_prompts)
        except:
            text_features = []
            batch_size= 1000
            for bi in range(text_prompt.shape[0]//batch_size):
                batch_text_features = self.encode_text(text_prompt[bi*1000:(bi+1)*1000], tokenized_prompts[bi*1000:(bi+1)*1000])
                text_features.append(batch_text_features)
            text_features = torch.cat(text_features, dim=0)
        n_dim = text_features.shape[-1]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(self.n_class, -1)

        self.text_features = text_features
        self.text_prompt = text_prompt
        self.tokenized_prompts = tokenized_prompts

    @property
    def dtype(self):
        return self.encode_image.conv1.weight.dtype


class DQ_QACoOp():
    def __init__(self, args, qclip, t_text_features=None, n_ctx=16, use_float32=False, use_grad_checkpoint=False):
        self.qclip = qclip
        self.use_grad_checkpoint = use_grad_checkpoint

        self.n_ctx = n_ctx # n_ctx 输入词数
        self.coop_lr = args.coop_lr*args.train_batch/20 # default lr scaling
        self.adapter_lr = args.adapter_lr
        self.wd = args.wd # weight_decay
        self.coop_epochs = args.coop_epochs
        self.adapter_epochs = args.adapter_epochs
        self.train_batch = args.train_batch 
        self.args = args
        self.t_text_features = t_text_features
        self.device = torch.device('cuda')
        
    def init_model(self, class_names, per_epoch_steps):
        self.t_clip = deepcopy(self.qclip)
        a_lp_qclip = deepcopy(self.qclip) # learnable prompts + quant clip: student

        self.a_lp_qclip = CLIP(args=self.args, 
                               class_names=class_names, 
                               clip_model=a_lp_qclip, 
                               n_ctx=self.n_ctx, 
                               cfg=self.qclip.cfg,
                               device=self.device).to(self.device) # transform quant clip: add learnable prompt
        
        if self.use_grad_checkpoint:
            try:
                self.a_lp_qclip.encode_text.transformer.use_gradient_checkpoint = True 
            except:
                self.a_lp_qclip.encode_text.module.transformer.use_gradient_checkpoint = True
        
        # Turn off gradients other than coop or adapter...
        for name, param in self.a_lp_qclip.named_parameters():
            if ('adapter' not in name) and ('prompt_learner' not in name):
                param.requires_grad_(False)
            # print('name:{} \t in adapter:{} \t  in prompt_learner:{} \t requires grad:{} \t'.format(name,'adapter' in name, 'prompt_learner' in name, param.requires_grad))
            # name:adapter.fc.0.weight         in adapter:True          in prompt_learner:False        requires grad:True 
            # name:adapter.fc.2.weight         in adapter:True          in prompt_learner:False        requires grad:True 
            # name:prompt_learner.ctx          in adapter:False         in prompt_learner:True         requires grad:True
        
        # coop
        param_dict = [{'params': [p for p in self.a_lp_qclip.prompt_learner.parameters() if p.requires_grad]}]
        self.optimizer_coop = torch.optim.SGD(param_dict, lr=self.coop_lr, weight_decay=self.wd)
        self.scheduler_coop = build_cosine_scheduler(
            self.optimizer_coop,
            lr=self.coop_lr,
            total_step=self.coop_epochs*per_epoch_steps)
        
        # adapter
        self.optimizer_adapter = torch.optim.AdamW(self.a_lp_qclip.adapter.parameters(), lr=self.adapter_lr, eps=1e-4)
        self.scheduler_adapter = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_adapter, self.adapter_epochs*per_epoch_steps)      
        
        if not(self.args.resume_checkpoint == 0):
            print(f'Resuming from checkpoint {self.args.resume_checkpoint}...')
            pl_path, ad_path = resolve_checkpoint_paths(self.args.load_dir, self.args.resume_checkpoint)
            load_params(self.a_lp_qclip.prompt_learner, pl_path)
            load_params(self.a_lp_qclip.adapter, ad_path)
            
        self.t_clip.model_dequant() # teacher

    def fit(self, data):
        train_loader = data['train_loader']
        
        if len(train_loader.dataset) < self.train_batch:
            real_img_bsz = len(train_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.train_batch 
        else:
            real_img_bsz = self.train_batch
        # pdb.set_trace()
        per_epoch_steps = len(train_loader)
        self.init_model(class_names=data['class_names'], per_epoch_steps=per_epoch_steps)

        self.t_clip.eval()
                                                                   
        print(f'Training P4Q from {self.args.resume_checkpoint} to {self.coop_epochs} epoch...')
        for epoch in range(self.args.resume_checkpoint, self.coop_epochs):
            for idx, (x, y) in enumerate(train_loader):
                cur_iter_idx = epoch*per_epoch_steps+idx # how many batch
                self.scheduler_coop.step(cur_iter_idx)
                
                # loss 1: distill output
                t_image_features =  self.t_clip.encode_image(x.to(self.device)).detach() # with torch.no_grad() the same
                t_image_features /= t_image_features.norm(dim=-1, keepdim=True)
                logit_scale = self.a_lp_qclip.logit_scale.exp()
                t_output = logit_scale * t_image_features @ self.t_text_features.t() # t_output
                
                s_output = self.a_lp_qclip(x.to(self.device)) # s_output
                
                # promblem 1: t_output cosine similarity, not 0-1, cross_entropy neg
                # ref 1: softmax distribution: https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb
                t_probs = t_output.softmax(dim=-1) # soft label
                s_probs = s_output.softmax(dim=-1)
                dis_weight = 1.
                loss_output_dis = F.cross_entropy(t_probs, s_probs) * dis_weight
             
                # loss 2: learn student $ adapter
                s_weight = 1.
                loss_student = F.cross_entropy(s_output, y.to(self.device)) * s_weight
                loss = loss_output_dis + loss_student 

                # optimize
                self.optimizer_coop.zero_grad()
                self.optimizer_adapter.zero_grad()
                
                loss.backward() 
                self.optimizer_coop.step() # coop
                self.optimizer_adapter.step() # adapter
                self.scheduler_adapter.step()
               
                if idx % self.args.print_freq == 0:
                    print('Epoch:{}/{}\t'
                          'Iter:{}/{}\t'
                          'loss_output_dis:{loss_output_dis:.3f}\t'
                          'loss_student:{loss_student:.3f}\t' 
                          'loss:{loss: .3f}'.format( 
                                epoch, self.coop_epochs, idx, per_epoch_steps, 
                                loss_output_dis=loss_output_dis, 
                                loss_student=loss_student,
                                loss=loss))
             
                if (epoch == self.coop_epochs-1) & (idx == int(per_epoch_steps * 0.9)):
                    self.a_lp_qclip.adapter.model_open_calibrate()
                if (epoch == self.coop_epochs-1) & (idx == per_epoch_steps-2):
                    self.a_lp_qclip.adapter.model_open_last_calibrate()

            # save model parameters
            if (epoch + 1) % self.args.save_freq == 0:
                save_dir = self.args.save_path
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                save_params(self.a_lp_qclip.prompt_learner , os.path.join(save_dir, f'epoch_{epoch+1}_prompt_learner.pth'))
                save_params(self.a_lp_qclip.adapter, os.path.join(save_dir, f'epoch_{epoch+1}_adapter.pth'))
                if epoch + 1 == self.coop_epochs:
                    save_params(self.a_lp_qclip.prompt_learner, os.path.join(save_dir, 'p4q_prompt_learner.pth'))
                    save_params(self.a_lp_qclip.adapter, os.path.join(save_dir, 'p4q_adapter.pth'))
        
        # import pdb
        # pdb.set_trace()
        if self.args.resume_checkpoint == self.coop_epochs:
            for idx, (x, y) in enumerate(train_loader):
                self.a_lp_qclip.adapter.model_open_calibrate()
                self.a_lp_qclip.adapter.model_open_last_calibrate()
                s_output = self.a_lp_qclip(x.to(self.device)) # s_output
                break
             
        self.a_lp_qclip.adapter.model_close_calibrate()
        self.a_lp_qclip.adapter.model_close_last_calibrate()
        self.a_lp_qclip.adapter.model_quant()
        self.a_lp_qclip.eval()
        self.a_lp_qclip.set_classifier() # final prompt text features
  
    @torch.no_grad()
    def inference(self,image):
        logits = self.a_lp_qclip(image, test=True)
        return logits.float().softmax(dim=-1)

    @torch.no_grad()
    def accuracy(self, loader, mean_per_class=False):
        if mean_per_class:
            return self._accuracy_mpc(loader)
        else:
            return self._accuracy(loader)

    def _accuracy_mpc(self, loader):
        n_class = self.n_class
        acc_per_class = [0 for _ in range(n_class)]
        count_per_class = [0 for _ in range(n_class)]
        for i, (x, y) in enumerate(loader):
            pred_y = self.inference(x.cuda())
            _, top_labels = pred_y.topk(1, dim=-1)
            for c in range(n_class):
                acc_per_class[c] += ((top_labels.view(-1) == y.cuda()) * (y.cuda()== c)).sum().item()
                count_per_class[c] += (y.cuda() == c).sum().item()
        acc = [a*1.0/c for (a, c) in zip(acc_per_class, count_per_class)]
        acc = np.array(acc).mean()
        return acc

    def _accuracy(self, loader):
        total_count=0
        acc_count =0
        for i,(x, y) in enumerate(loader):
            pred_y = self.inference(x.cuda())
            _, top_labels = pred_y.topk(1, dim=-1)
            acc_count += (top_labels.view(-1)==y.cuda()).sum().cpu().numpy()
            total_count += y.shape[0]
        acc = acc_count*1.0/total_count
        acc = acc.item()
        return acc
