# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, send_model_cuda
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.semireward import Rewarder, Generator, EMARewarder, cosine_similarity_n


@ALGORITHMS.register('srfixmatch')
class SRFixMatch(AlgorithmBase):

    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).
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
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
        self.N_k = args.N_k
        # self.rewarder = (Rewarder(128, args.feature_dim).cuda(device=args.gpu) if args.sr_ema == 0 \
        #                  else EMARewarder(128, feature_dim=args.feature_dim, ema_decay=args.sr_ema_m).cuda(device=args.gpu))
        # self.generator = Generator(args.feature_dim).cuda(device=args.gpu)
        self.rewarder = send_model_cuda(args, Rewarder(128, args.feature_dim)) if args.sr_ema == 0 \
                        else send_model_cuda(args, EMARewarder(128, feature_dim=args.feature_dim, ema_decay=args.sr_ema_m), clip_batch=False)
        self.generator = send_model_cuda(args, Generator(args.feature_dim))
        self.start_timing = args.start_timing

        self.rewarder_optimizer = torch.optim.Adam(self.rewarder.parameters(), lr=args.sr_lr)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.sr_lr)

        self.criterion = torch.nn.MSELoss()

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def data_generator(self, x_lb, y_lb, x_ulb_w, x_ulb_s, rewarder,gpu):
        gpu = gpu
        rewarder = rewarder.eval()
        unsup_losss = None
        rewards = None
        for _ in range(self.sr_decay()):  
            num_lb = y_lb.shape[0]
            with self.amp_cm():
                if self.use_cat:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                    outputs = self.model(inputs)
                    logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                    feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
                else:
                    outs_x_ulb_s = self.model(x_ulb_s)
                    logits_x_ulb_s = outs_x_ulb_s['logits']
                    feats_x_ulb_s = outs_x_ulb_s['feat']
                    with torch.no_grad():
                        outs_x_ulb_w = self.model(x_ulb_w)
                        logits_x_ulb_w = outs_x_ulb_w['logits']
                        feats_x_ulb_w = outs_x_ulb_w['feat']

                probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                      logits=probs_x_ulb_w,
                                      use_hard_label=self.use_hard_label,
                                      T=self.T,
                                      softmax=False)
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            reward = rewarder(feats_x_ulb_w, pseudo_label)
            reward = reward.mean()
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label,'ce', mask=mask)

            rewards = torch.cat((rewards, reward.unsqueeze(0))) if rewards is not None else reward.unsqueeze(0)
            unsup_losss = torch.cat((unsup_losss, unsup_loss.unsqueeze(0))) if unsup_losss is not None else unsup_loss.unsqueeze(0)
        avg_rewards = rewards.mean()
        mask = torch.where(rewards > avg_rewards, torch.tensor(1).cuda(gpu), torch.tensor(0).cuda(gpu))
        unsup_loss = unsup_loss * mask
        unsup_loss = unsup_loss.mean()

        return unsup_loss

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            # SemiReward inference
            if self.it > self.start_timing:
                rewarder = self.rewarder
                unsup_loss = self.data_generator(x_lb, y_lb, x_ulb_w, x_ulb_s,rewarder,self.gpu)
            else:
                unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)

            # SemiReward training
            if self.it > 0:
                # Generate pseudo labels using the generator (your pseudo-labeling process)
                self.rewarder.train()
                self.generator.train()
                generated_label = self.generator(feats_x_lb).detach()

                # Convert generated pseudo labels and true labels to tensors
                real_labels_tensor = y_lb.cuda(self.gpu).view(-1)
                real_labels_tensor=real_labels_tensor.unsqueeze(0)

                generated_label = generated_label.view(-1)               
                reward = self.rewarder(feats_x_lb,generated_label.long())
                reward = reward.mean().unsqueeze(0)

                if self.it >= self.start_timing:
                    filtered_pseudo_labels = pseudo_label
                    filtered_feats_x_ulb_w = feats_x_ulb_w
                    rewarder = self.rewarder.eval()

                    max_reward = -float('inf')
                    reward = self.rewarder(feats_x_ulb_w, pseudo_label)
                    reward = reward.mean()

                    max_reward = torch.where(reward > max_reward, reward, max_reward)
                    filtered_pseudo_labels = torch.where(reward > max_reward, pseudo_label, filtered_pseudo_labels)
                    filtered_feats_x_ulb_w = torch.where(reward > max_reward, feats_x_ulb_w, filtered_feats_x_ulb_w)
                    if self.it % self.N_k == 0 and self.it > self.start_timing:
                        max_reward = -float('inf')

                        self.rewarder.train()
                        self.generator.train()
                        generated_label = self.generator(feats_x_ulb_w).detach()
                        generated_label = generated_label.view(-1)
                        reward = self.rewarder(feats_x_ulb_w, generated_label.long())
                        reward = reward.mean()
                        cosine_similarity_score = cosine_similarity_n(torch.floor(generated_label), filtered_pseudo_labels)
                        generator_loss = self.criterion(reward, torch.ones_like(reward).cuda(self.gpu))
                        rewarder_loss = self.criterion(reward, cosine_similarity_score)

                        generator_loss.backward(retain_graph=True)
                        rewarder_loss.backward(retain_graph=True)

                        self.generator_optimizer.step()
                        self.rewarder_optimizer.step()

                        self.generator_optimizer.zero_grad()
                        self.rewarder_optimizer.zero_grad()
                else:
                    cosine_similarity_score = cosine_similarity_n(torch.floor(generated_label), real_labels_tensor) 
                    generator_loss = self.criterion(reward, torch.ones_like(reward).cuda(self.gpu))
                    rewarder_loss = self.criterion(reward, cosine_similarity_score)

                    generator_loss.backward(retain_graph=True)
                    rewarder_loss.backward(retain_graph=True)

                    self.generator_optimizer.step()
                    self.rewarder_optimizer.step()

                    self.generator_optimizer.zero_grad()
                    self.rewarder_optimizer.zero_grad()

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--start_timing', int,20000),
            SSL_Argument('--feature_dim', int,384),
            SSL_Argument('--sr_lr', float, 0.0005),
            SSL_Argument('--N_k', int, 10),
            SSL_Argument('--sr_ema', str2bool, True),
            SSL_Argument('--sr_ema_m', float, 0.999),
        ]
