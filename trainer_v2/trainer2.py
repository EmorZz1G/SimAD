from trainer_v2.exp import Trainer
import torch
import torch.nn as nn
import os
import numpy as np
import time
import math
from sklearn.preprocessing import minmax_scale
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support

def my_best_f1(score, label):
    best_f1 = (0,0,0)
    best_thre = 0
    # score = minmax_scale(score)
    # for q in np.arange(0.001, 0.501, 0.001):
    for q in np.arange(0.01, 0.901, 0.01):
        thre = np.quantile(score, 1-q)
        pred = score > thre
        pred = pred.astype(int)
        label = label.astype(int)
        p,r,f1,_ = precision_recall_fscore_support(label, pred, average='binary')
        # print(f'q: {q}, p: {p}, r: {r}, f1: {f1}')

        if f1 > best_f1[2]:
            best_f1 = (p, r, f1)
            best_thre = thre

    return *best_f1, best_thre


def rec_score_func2(x_out, x_patch, c_dim):
    from torch.nn.functional import interpolate
    l2_loss = nn.MSELoss(reduction='none')
    cos_loss = nn.CosineSimilarity(dim=-1)
    l2_score = l2_loss(x_out, x_patch)
    l2_score = rearrange(l2_score, 'b l (p c) -> b (l p) c', c=c_dim).mean(-1)
    seq_len = l2_score.shape[1]
    cos_score = 1 - cos_loss(x_out, x_patch)
    cos_score = interpolate(cos_score.unsqueeze(1), size=seq_len, mode='linear', align_corners=False).squeeze(1)
    score = l2_score + cos_score
    return score

import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from einops import rearrange


def patch_mask_aug(x, patch_size, mask_ratio=0.1):
    x = x.clone()
    B, L, C = x.shape
    x_patch = rearrange(x,'b (l p) c -> b l (p c)', p=patch_size)
    B, N, P = x_patch.shape
    mask_patch_num = 1
    idx = torch.randperm(N)[:mask_patch_num]
    x_patch[:,idx,:] = -1
    x = rearrange(x_patch,'b l (p c) -> b (l p) c', p=patch_size, c=C)
    return x, idx

def patch_shuffle_aug(x, patch_size):
    x = x.clone()
    B, L, C = x.shape
    x_patch = rearrange(x,'b (l p) c -> b l (p c)', p=patch_size)
    B, N, P = x_patch.shape
    idx = torch.randperm(N)
    x_patch = x_patch[:,idx,:]
    x = rearrange(x_patch,'b l (p c) -> b (l p) c', p=patch_size, c=C)
    return x


def aug_noise(x, patch_size, noise_level=0.1):
    x = x.clone()
    B, L, C = x.shape
    noise = torch.randn_like(x) * noise_level
    x = x + noise
    # patch_x = rearrange(x,'b (l p) c -> b l (p c)', p=patch_size)
    return x

from trainer_v2.mylogger import MyLogger

from metrics.combine_all_scores import combine_all_evaluation_scores, combine_all_evaluation_scores_with_bias

class ContAD_Trainer(Trainer):
    def __init__(self, config, model, get_loader_func):
        super().__init__(config, model, get_loader_func)


        if config.resume==0:
            self.mylogger = MyLogger(config, model)
            print(f'==============logger id {self.mylogger.get_id}==========')
            print(f'==============logger id {self.mylogger.get_id}==========')

        self.cur_epoch = 0
        self.cur_iter = 0

        self.topf1 = []

    def train(self):
        
        l2_loss = nn.MSELoss()
        cos_loss = nn.CosineSimilarity(dim=-1)
        bce_loss = nn.BCEWithLogitsLoss()
        dataloader = self.train_loader
        opti = self.optimizer
        model = self.model
        iters = 0
        device = self.device
        st = time.time()

        # lr_func = lambda step: min(
        #     (step + 1) / (warmup_steps + 1e-8),
        #     0.5 * (math.cos(step / self.config.epochs * math.pi) + 1),
        # )

        max_iters = 150000
        scheduler = CosineLRScheduler(opti,
                                  t_initial=max_iters,
                                  lr_min=self.config.lr*0.01,
                                  warmup_lr_init=self.config.lr*0.001,
                                  warmup_t=max_iters//10,
                                  cycle_limit=1,
                                  t_in_epochs=False,
                                 )

        if self.config.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for e in range(self.config.epochs):
            model.train()
            e_st = time.time()
            data_iters = 0
            self.cur_epoch = e
            for i, (x, y) in enumerate(dataloader):
                data_iters += 1

                if data_iters > 500:
                    break
                
                x = x.to(device)
                x_patch = rearrange(x,'b (l p) c -> b l (p c)', p=model.patch_size)

                x_out, sim_score = model(x)
                rec_loss = l2_loss(x_out, x_patch) + torch.mean(1 - cos_loss(x_out, x_patch))

                x2 = aug_noise(x, model.patch_size, self.config.noise_level)
                x_out2, sim_score2 = model(x2)
                x_patch2 = rearrange(x2,'b (l p) c -> b l (p c)', p=model.patch_size)
                rec_loss2 = l2_loss(x_out2, x_patch) + torch.mean(1 - cos_loss(x_out2, x_patch))
                

                warmup_steps = self.config.warmup_steps
                warmup_max_ratio = self.config.warmup_max_ratio
                warmup = min((iters + 1) / (warmup_steps + 1e-8), warmup_max_ratio)
                loss = rec_loss + rec_loss2 

                sim_score = F.normalize(sim_score, dim=-1)
                sim_score2 = F.normalize(sim_score2, dim=-1)

                sim_loss1 = l2_loss(sim_score, sim_score2.detach()) + torch.mean(1 - cos_loss(sim_score, sim_score2.detach()))
                sim_loss2 = l2_loss(sim_score2, sim_score.detach()) + torch.mean(1 - cos_loss(sim_score2, sim_score.detach()))

                # sim_loss1 = l2_loss(sim_score, sim_score2) + torch.mean(1 - cos_loss(sim_score, sim_score2))
                # sim_loss2 = l2_loss(sim_score2, sim_score) + torch.mean(1 - cos_loss(sim_score2, sim_score))
                
                sim_loss = sim_loss1 + sim_loss2

                loss -= sim_loss * warmup


                # only for ablation study
                # loss = rec_loss
                # rec_loss2 = sim_loss = sim_loss1 = sim_loss2 = torch.zeros(1).to(device)

                if self.config.use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()



                self.cur_iter = iters
                if iters % self.config.print_freq == 0:
                    print('Epoch: {}, Iter: {} lr: {:.8f} Loss: {:.4f}'.format(e, iters, opti.param_groups[0]['lr'], loss.item()))
                    print('Epoch: {}, Iter: {}, Rec1: {:.4f}, Time: {:.4f}'.format(e, iters, rec_loss.item(), time.time()-e_st))
                    print('Epoch: {}, Iter: {}, Rec2: {:.4f}, Time: {:.4f}'.format(e, iters, rec_loss2.item(), time.time()-e_st))
                    # print('Epoch: {}, Iter: {}, Rec3: {:.4f}, Time: {:.4f}'.format(e, iters, rec_loss3.item(), time.time()-e_st))
                    print('Epoch: {}, Iter: {}, Sim1: {:.4f}, Time: {:.4f}'.format(e, iters, sim_loss.item(), time.time()-e_st))

                    self.mylogger.add_iter_value({
                        'epoch': e,
                        'iter': iters,
                        'loss': loss.item(),
                        'rec1': rec_loss.item(),
                        'rec2': rec_loss2.item(),
                        'sim1': sim_loss1.item(),
                        'sim2': sim_loss2.item(),
                        'sim_loss': sim_loss2.item(),
                        'lr':  opti.param_groups[0]['lr'],
                    })


                


                if ((i + 1) % self.config.accumulate_grad_steps == 0) or (i + 1 == len(dataloader)):
                    if self.config.use_amp:
                        # nn.utils.clip_grad_norm_(model.parameters(), 1)
                        scaler.step(opti)
                        scaler.update()
                        opti.step()
                        opti.zero_grad()

                    else:
                        # nn.utils.clip_grad_norm_(model.parameters(), 1)
                        opti.step()
                        opti.zero_grad()

                if iters % self.config.test_freq == 0:
                    model.eval()
                    self.test(0)
                    torch.save(model, os.path.join(self.save_pth, 'checkout.pth'))
                    self.mylogger.save_all()
                    model.train()

                scheduler.step_update(i)
                iters += 1


                
        print('='*20+'Training Finished'+'='*20)
        print('Total Time cost: {:.4f}'.format(time.time()-st))

            

    @torch.no_grad()
    def test(self, from_file=0,is_best=0):
        if from_file:
            # model = torch.load(os.path.join(self.save_pth, 'checkout.pth'))
            find_model=False
            print('try to find best model')
            print(self.best_pth)

            if is_best:
                model = torch.load(os.path.join(self.logs_pth, 'best_checkout.pth'),map_location=self.device).to(self.device)
            else:
                if self.best_pth.find('none') == -1:
                    try:
                        print(self.best_pth)
                        model = torch.load(os.path.join(self.best_pth, 'best_checkout.pth'),map_location=self.device).to(self.device)
                        find_model=True
                    except:
                        find_model=False
                        print('No best model found')
                    
                

                if not find_model:
                    tmp = self.best_pth.replace('none', '')

                    file_cnt_max = 0
                    tar_file = ""
                    for file2 in os.listdir(tmp):
                        # 计算文件夹里面的文件个数
                        file_cnt = len(os.listdir(os.path.join(tmp, file2)))
                        if file_cnt > file_cnt_max:
                            file_cnt_max = file_cnt
                            tar_file = file2

                    print(f'find model in {tar_file}')
                    try:
                        model = torch.load(os.path.join(tmp, tar_file, 'best_checkout.pth'),map_location=self.device).to(self.device)
                        find_model=True
                    except:
                        print(os.path.join(tmp, tar_file, 'best_checkout.pth'))
                        print('No model found , when use last model ???')
                        find_model=False
                    

                if not find_model:
                    try:
                        tmp = self.save_pth.replace('none', 'saved_models')
                        print(tmp)
                        model = torch.load(os.path.join(tmp, 'checkout.pth'),map_location=self.device).to(self.device)
                        find_model=True
                    except:
                        print('No model found , when use last model')
                        find_model=False
                        return



        else:
            model = self.model

        model.eval()
        dataloader = self.test_loader
        device = self.device


        labels = []
        scores = []
        st = time.time()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            x_out, sim_score = model(x)

            x_patch = rearrange(x,'b (l p) c -> b l (p c)', p=model.patch_size)

            rec_score = rec_score_func2(x_out, x_patch, model.c_dim)

            score = rec_score
            # score =  sim_score

            labels.append(y.view(-1).cpu())
            scores.append(score.view(-1).cpu())

        print('Test time cost: {:.4f}'.format(time.time()-st))
        labels = np.concatenate(labels ,axis=0)
        scores = np.concatenate(scores, axis=0)


        p,r,f1,shr = my_best_f1(scores, labels)
        print(f'sh {shr} p: {p}, r: {r}, f1: {f1}')

        pred = scores > shr
        labels = labels.astype(int)
        pred = pred.astype(int)

        if from_file:
            metrics = combine_all_evaluation_scores_with_bias(pred,labels,scores,full=True,data_name=self.config.data_name)

            print('====================Test Full====================')
            for key, value in metrics.items():
                # 输出数据
                print('{0:21} : {1:0.4f}'.format(key, value))

            metrics['model_id'] = self.config.model_id
            metrics['index'] = self.config.index
            file_ = os.path.join(self.logs_pth, self.config.data_name+'_v1g.csv')
            if not os.path.exists(file_):
                pd.DataFrame([metrics]).to_csv(file_, index=True)
            else:
                datas = pd.read_csv(file_, index_col=0)
                if datas[datas['model_id'] == self.config.model_id].shape[0] == 0:
                    datas = datas._append(metrics, ignore_index=True)
                    pd.DataFrame(datas).to_csv(file_, index=True)
                
            
        save_all=0
        if len(self.topf1)>0 and f1 > self.topf1[-1][0] and self.config.resume==0 and save_all:
            metrics = combine_all_evaluation_scores_with_bias(pred,labels,scores,full=True,data_name=self.config.data_name)

            print('====================Test Full====================')
            for key, value in metrics.items():
                # 输出数据
                print('{0:21} : {1:0.4f}'.format(key, value))

            metrics['model_id'] = self.config.model_id
            metrics['index'] = self.config.index
            pth = self.mylogger.save_pth
            file_ = os.path.join(pth, self.config.data_name+'_test_metrics.csv')
            if not os.path.exists(file_):
                pd.DataFrame([metrics]).to_csv(file_, index=True)
            else:
                datas = pd.read_csv(file_, index_col=0)
                if datas[datas['model_id'] == self.config.model_id].shape[0] == 0 or self.model_id == 'none':
                    datas = datas._append(metrics, ignore_index=True)
                    pd.DataFrame(datas).to_csv(file_, index=True)

        self.topf1.append((f1,p,r))
        self.topf1.sort(key=lambda x:x[0], reverse=True)
        for i in range(len(self.topf1)):
            if i>=3:break
            print(f'{i+1}: {self.topf1[i]}')


        metric_score = np.sum(scores)
        print('metric_score', metric_score)
        if self.config.resume==0:
            self.mylogger.add_iter_value({
                'epoch': self.cur_epoch,
                'iter': self.cur_iter,
                'f1':f1,
                'pre':p,
                'rec':r,
                # 'quantile':shr,
                'thres':shr,
                'metric_score':metric_score
            }, train=0, metric='metric_score', max_=0)

        