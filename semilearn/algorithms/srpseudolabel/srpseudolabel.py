# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, send_model_cuda
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument,str2bool
from semilearn.algorithms.semireward import Rewarder, Generator, EMARewarder, cosine_similarity_n, add_gaussian_noise, label_dim

@ALGORITHMS.register('srpseudolabel')
class SRPseudoLabel(AlgorithmBase):
    """
        Pseudo Label algorithm (https://arxiv.org/abs/1908.02983).
        SemiReward algorithm (https://arxiv.org/abs/2310.03013).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(p_cutoff=args.p_cutoff, unsup_warm_up=args.unsup_warm_up)
        self.task_type = args.task_type
        self.N_k = args.N_k
        self.rewarder = send_model_cuda(args, Rewarder(label_dim(self.num_classes), 128, args.feature_dim)) if args.sr_ema == 0 \
                        else send_model_cuda(args, EMARewarder(label_dim(self.num_classes), 128, feature_dim=args.feature_dim, ema_decay=args.sr_ema_m), clip_batch=False)
        self.generator = send_model_cuda(args, Generator(args.feature_dim))
        self.start_timing = args.start_timing

        self.rewarder_optimizer = torch.optim.Adam(self.rewarder.parameters(), lr=args.sr_lr)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.sr_lr)

        self.criterion = torch.nn.MSELoss()

    def init(self, p_cutoff, unsup_warm_up=0.4):
        self.p_cutoff = p_cutoff
        self.unsup_warm_up = unsup_warm_up 

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def data_generator(self, x_lb, x_ulb_w, rewarder,gpu):
        gpu = gpu
        rewarder = rewarder.eval()
        for _ in range(self.sr_decay()):  
            with self.amp_cm():

                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']

            # calculate BN only for the first batch
                self.bn_controller.freeze_bn(self.model)
                if self.task_type == 'cls':
                    outs_x_ulb = self.model(x_ulb_w)
                    logits_x_ulb = outs_x_ulb['logits']
                else:
                    noisy_x_ulb_w = add_gaussian_noise(x_ulb_w, mean=0, std=1)
                    outs_x_ulb = self.model(noisy_x_ulb_w)
                    outs_x_ulb_pseudo = self.model(x_ulb_w)
                    logits_x_ulb = outs_x_ulb['logits']
                    outs_x_ulb_pseudo = outs_x_ulb_pseudo['logits']
                feats_x_ulb = outs_x_ulb['feat']
                self.bn_controller.unfreeze_bn(self.model)
            # compute mask
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb.detach())

            # generate unlabeled targets using pseudo label hook
                pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb.detach() if self.task_type == 'cls' else outs_x_ulb_pseudo,
                                          use_hard_label=True)
            reward = rewarder(feats_x_ulb, pseudo_label)
            avg_reward=reward.mean()
            mask2 = torch.where(reward >= avg_reward, torch.tensor(1).cuda(gpu), torch.tensor(0).cuda(gpu)).squeeze().float()
            unsup_loss = self.consistency_loss(logits_x_ulb, pseudo_label,'ce', mask=mask,mask2=mask2)
        unsup_loss = unsup_loss

        return unsup_loss

    def train_step(self, x_lb, y_lb, x_ulb_w):
        # inference and calculate sup/unsup losses
        with self.amp_cm():

            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']

            # calculate BN only for the first batch
            self.bn_controller.freeze_bn(self.model)
            if self.task_type == 'cls':
                outs_x_ulb = self.model(x_ulb_w)
                logits_x_ulb = outs_x_ulb['logits']
            else:
                noisy_x_ulb_w = add_gaussian_noise(x_ulb_w, mean=0, std=1)
                outs_x_ulb = self.model(noisy_x_ulb_w)
                outs_x_ulb_pseudo = self.model(x_ulb_w)
                logits_x_ulb = outs_x_ulb['logits']
                outs_x_ulb_pseudo = outs_x_ulb_pseudo['logits']
            feats_x_ulb = outs_x_ulb['feat']
            self.bn_controller.unfreeze_bn(self.model)

            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb.detach())

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb.detach() if self.task_type == 'cls' else outs_x_ulb_pseudo,
                                          use_hard_label=True)
            # SemiReward inference
            if self.it > self.start_timing:
                rewarder = self.rewarder
                unsup_loss = self.data_generator(x_lb, x_ulb_w, rewarder, self.gpu)
            else:
                unsup_loss = self.consistency_loss(logits_x_ulb, pseudo_label,
                                               name='ce' if self.task_type == 'cls' else 'l1',
                                               mask=mask)

            # SemiReward training
            if self.it > 0:
            # Generate pseudo labels using the generator (your pseudo-labeling process)
                self.rewarder.train()
                self.generator.train()
                generated_label = self.generator(feats_x_lb.detach()).detach()
                generated_label=generated_label.long()
            # Convert generated pseudo labels and true labels to tensors
                real_labels_tensor = y_lb.cuda(self.gpu)          
                reward = self.rewarder(feats_x_lb.detach(),generated_label.squeeze(1))
                if self.it >= self.start_timing:
                    filtered_pseudo_labels = pseudo_label.long()
                    filtered_feats_x_ulb_w = feats_x_ulb.detach()
                    rewarder = self.rewarder.eval()
                    max_reward = -float('inf')
                    reward = self.rewarder(feats_x_ulb.detach(), pseudo_label.long())
                    reward = reward.mean()
                    max_reward = torch.where(reward > max_reward, reward, max_reward)
                    filtered_pseudo_labels = torch.where(reward > max_reward, pseudo_label.detach(), filtered_pseudo_labels)
                    filtered_feats_x_ulb_w = torch.where(reward > max_reward, feats_x_ulb.detach(), filtered_feats_x_ulb_w)
                    if self.it % self.N_k == 0 and self.it > self.start_timing:
                        self.rewarder.train()
                        self.generator.train()
                        generated_label = self.generator(filtered_feats_x_ulb_w.squeeze(1)).detach()
                        generated_label=generated_label.long()
                        reward = self.rewarder(filtered_feats_x_ulb_w, generated_label.squeeze(1))
                        generated_label = F.one_hot(generated_label.squeeze(1), num_classes=self.num_classes)
                        filtered_pseudo_labels= F.one_hot(filtered_pseudo_labels.long(), num_classes=self.num_classes)
                        cosine_similarity_score = cosine_similarity_n(generated_label.float(), filtered_pseudo_labels.float())
                        generator_loss = self.criterion(reward, torch.ones_like(reward).cuda(self.gpu))
                        rewarder_loss = self.criterion(reward, cosine_similarity_score)

                        self.generator_optimizer.zero_grad()
                        self.rewarder_optimizer.zero_grad()
                
                        generator_loss.backward(retain_graph=True)
                        rewarder_loss.backward(retain_graph=True)

                        self.generator_optimizer.step()
                        self.rewarder_optimizer.step()
                else:
                    generated_label = F.one_hot(generated_label.squeeze(1), num_classes=self.num_classes)
                    real_labels_tensor=F.one_hot(real_labels_tensor, num_classes=self.num_classes)
                    cosine_similarity_score = cosine_similarity_n(generated_label.float(), real_labels_tensor.float()) 
                    generator_loss = self.criterion(reward, torch.ones_like(reward).cuda(self.gpu))
                    rewarder_loss = self.criterion(reward, cosine_similarity_score)

                    self.generator_optimizer.zero_grad()
                    self.rewarder_optimizer.zero_grad()
                
                    generator_loss.backward(retain_graph=True)
                    rewarder_loss.backward(retain_graph=True)

                    self.generator_optimizer.step()
                    self.rewarder_optimizer.step()

            # calculate unlabeled loss
            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
            SSL_Argument('--task_type', str, 'cls'),
            # SSL_Argument('--use_flex', str2bool, False),
            SSL_Argument('--start_timing', int,20000),
            SSL_Argument('--feature_dim', int,384),
            SSL_Argument('--sr_lr', float, 0.0005),
            SSL_Argument('--N_k', int, 10),
            SSL_Argument('--sr_ema', str2bool, True),
            SSL_Argument('--sr_ema_m', float, 0.999),
        ]