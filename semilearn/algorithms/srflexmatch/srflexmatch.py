
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from .utils import FlexMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, send_model_cuda
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.semireward import Rewarder, Generator, EMARewarder, cosine_similarity_n, label_dim


@ALGORITHMS.register('srflexmatch')
class SRFlexMatch(AlgorithmBase):
    """
        FlexMatch algorithm (https://arxiv.org/abs/2110.08263).
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
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # flexmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)
        self.N_k = args.N_k
        self.rewarder = send_model_cuda(args, Rewarder(label_dim(self.num_classes), 128, args.feature_dim)) if args.sr_ema == 0 \
                        else send_model_cuda(args, EMARewarder(label_dim(self.num_classes), 128, feature_dim=args.feature_dim, ema_decay=args.sr_ema_m), clip_batch=False)
        self.generator = send_model_cuda(args, Generator(args.feature_dim))
        self.start_timing = args.start_timing

        self.rewarder_optimizer = torch.optim.Adam(self.rewarder.parameters(), lr=args.sr_lr)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.sr_lr)

        self.criterion = torch.nn.MSELoss()

        self.max_reward = -float('inf')
    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FlexMatchThresholdingHook(
            ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes, thresh_warmup=self.args.thresh_warmup), "MaskingHook")
        super().set_hooks()

    def data_generator(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s, rewarder,gpu):
        gpu = gpu
        rewarder = rewarder.eval()
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
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb)
            reward = rewarder(feats_x_ulb_w, pseudo_label)
            avg_reward=reward.mean()
            mask2 = torch.where(reward >= avg_reward, torch.tensor(1).cuda(gpu), torch.tensor(0).cuda(gpu)).squeeze().float()
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label,'ce', mask=mask,mask2=mask2)
        unsup_loss = unsup_loss
        return unsup_loss


    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
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
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False, idx_ulb=idx_ulb)
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)
            if self.it > self.start_timing:
                rewarder = self.rewarder
                unsup_loss = self.data_generator(x_lb, y_lb,idx_ulb, x_ulb_w, x_ulb_s,rewarder,self.gpu)
            else:
                pseudo_label = pseudo_label
                unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label,'ce', mask=mask)
            
            if self.it > 0:
            # Generate pseudo labels using the generator (your pseudo-labeling process)
                self.rewarder.train()
                self.generator.train()
                generated_label = self.generator(feats_x_lb.detach())
                generated_label=generated_label.long()
            # Convert generated pseudo labels and true labels to tensors
                real_labels_tensor = y_lb.cuda(self.gpu)          
                reward = self.rewarder(feats_x_lb.detach(),generated_label.squeeze(1))
                if self.it >= self.start_timing:
                    filtered_pseudo_labels = pseudo_label.long()
                    filtered_feats_x_ulb_w = feats_x_ulb_w.detach()
                    rewarder = self.rewarder.eval()
                    
                    reward = self.rewarder(feats_x_ulb_w.detach(), pseudo_label.long())
                    reward = reward.mean()
                    self.max_reward = torch.where(reward > self.max_reward, reward, self.max_reward)
                    filtered_pseudo_labels = torch.where(reward > self.max_reward, pseudo_label.detach(), filtered_pseudo_labels)
                    filtered_feats_x_ulb_w = torch.where(reward > self.max_reward, feats_x_ulb_w.detach(), filtered_feats_x_ulb_w)
                    if self.it % self.N_k == 0 and self.it > self.start_timing:
                        self.max_reward = -float('inf')
                        self.rewarder.train()
                        self.generator.train()
                        generated_label = self.generator(filtered_feats_x_ulb_w.squeeze(1))
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

            total_loss = sup_loss + self.lambda_u * unsup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
            SSL_Argument('--start_timing', int,20000),
            SSL_Argument('--feature_dim', int,384),
            SSL_Argument('--sr_lr', float, 0.0005),
            SSL_Argument('--N_k', int, 10),
            SSL_Argument('--sr_ema', str2bool, True),
            SSL_Argument('--sr_ema_m', float, 0.999),
        ]
