import torch
import torch.nn.functional as F
from semilearn.algorithms.freematch.utils import FreeMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS, send_model_cuda
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.semireward import Rewarder, Generator, EMARewarder, cosine_similarity_n, label_dim


# TODO: move these to .utils or algorithms.utils.loss
def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


@ALGORITHMS.register('srfreematch')
class SRFreeMatch(AlgorithmBase):
    """
        FreeMatch algorithm (https://arxiv.org/abs/2205.07246).
        SemiReward algorithm (https://arxiv.org/abs/2310.03013).
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
        self.lambda_e = args.ent_loss_ratio
        self.N_k = args.N_k
        self.rewarder = send_model_cuda(args, Rewarder(label_dim(self.num_classes), 128, args.feature_dim)) if args.sr_ema == 0 \
                        else send_model_cuda(args, EMARewarder(label_dim(self.num_classes), 128, feature_dim=args.feature_dim, ema_decay=args.sr_ema_m), clip_batch=False)
        self.generator = send_model_cuda(args, Generator(args.feature_dim))
        self.start_timing = args.start_timing

        self.rewarder_optimizer = torch.optim.Adam(self.rewarder.parameters(), lr=args.sr_lr)
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=args.sr_lr)

        self.criterion = torch.nn.MSELoss()

    def init(self, T, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh

    def data_generator(self, x_lb, y_lb, x_ulb_w, x_ulb_s, rewarder,gpu):
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
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)
            reward = rewarder(feats_x_ulb_w, pseudo_label)
            avg_reward=reward.mean()
            mask2 = torch.where(reward >= avg_reward, torch.tensor(1).cuda(gpu), torch.tensor(0).cuda(gpu)).squeeze().float()
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label,'ce', mask=mask,mask2=mask2)
        unsup_loss = unsup_loss

        return unsup_loss

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholdingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
        super().set_hooks()

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

            # calculate mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            # SemiReward inference
            if self.it > self.start_timing:
                rewarder = self.rewarder
                unsup_loss = self.data_generator(x_lb, y_lb, x_ulb_w, x_ulb_s, rewarder, self.gpu)
            else:
                unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label,'ce', mask=mask)

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
                    filtered_feats_x_ulb_w = feats_x_ulb_w.detach()
                    rewarder = self.rewarder.eval()
                    max_reward = -float('inf')
                    reward = self.rewarder(feats_x_ulb_w.detach(), pseudo_label.long())
                    reward = reward.mean()
                    max_reward = torch.where(reward > max_reward, reward, max_reward)
                    filtered_pseudo_labels = torch.where(reward > max_reward, pseudo_label.detach(), filtered_pseudo_labels)
                    filtered_feats_x_ulb_w = torch.where(reward > max_reward, feats_x_ulb_w.detach(), filtered_feats_x_ulb_w)
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
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)

            # calculate entropy loss
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0
            # ent_loss = 0.0
            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
            SSL_Argument('--start_timing', int,20000),
            SSL_Argument('--feature_dim', int,384),
            SSL_Argument('--sr_lr', float, 0.0005),
            SSL_Argument('--N_k', int, 10),
            SSL_Argument('--sr_ema', str2bool, True),
            SSL_Argument('--sr_ema_m', float, 0.999),
        ]