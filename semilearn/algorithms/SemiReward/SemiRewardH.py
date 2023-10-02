import torch
from semilearn.algorithms.SemiReward import Rewarder,Generator,cosine_similarity_n
import numpy as np

class SemiReward_infer:
    def __init__(self, rewarder_model, starttiming):
        self.rewarder = rewarder_model
        self.starttiming = starttiming

    def __call__(self, feats_x_ulb_w, pseudo_label, it):
        pseudo_label_list = []
        if it >= self.starttiming:
            reward_list = []
            

            for _ in range(256):
                reward = self.rewarder(feats_x_ulb_w, pseudo_label)
                pseudo_label_list.append(pseudo_label)
                reward_list.append(reward.item())

            # Calculate the average reward
            average_reward = sum(reward_list) / len(reward_list)

            # Filter out pseudo_labels with rewards below the average
            filtered_pseudo_labels = []
            for i, reward in enumerate(reward_list):
                if reward >= average_reward:
                    filtered_pseudo_labels.append(pseudo_label_list[i])

            return filtered_pseudo_labels
        else:
            for _ in range(256):
                pseudo_label_list.append(pseudo_label)
            return pseudo_label_list

class SemiReward_train:
    def __init__(self, rewarder_model, generator_model, criterion, starttiming,gpu):
        self.rewarder = rewarder_model
        self.generator = generator_model
        self.criterion = criterion
        self.starttiming = starttiming
        self.gpu=gpu
    def __call__(self, feats_x_ulb_w, pseudo_label, y_lb, it):
        generated_label = self.generator(feats_x_ulb_w).detach()

        real_labels_tensor = y_lb.cuda(self.gpu).view(-1)
        real_labels_tensor=real_labels_tensor.unsqueeze(0)
        if it >= self.starttiming:
            accumulated_pseudo_labels = []  
            reward_list = []  

            for _ in range(80):
                reward = self.rewarder(feats_x_ulb_w, pseudo_label)
                accumulated_pseudo_labels.append(pseudo_label.squeeze().cpu().numpy())
                reward_list.append(reward.item())

            sorted_indices = np.argsort(reward_list)[-8:]
            filtered_pseudo_labels = [accumulated_pseudo_labels[i] for i in sorted_indices]
            reward  = [reward_list[i] for i in sorted_indices]

            filtered_pseudo_labels_tensor = torch.tensor(filtered_pseudo_labels, dtype=torch.float32).cuda(self.gpu)
            reward = torch.tensor(reward, dtype=torch.float32, requires_grad=True).cuda(self.gpu)


            cosine_similarity_score = cosine_similarity_n(generated_label, filtered_pseudo_labels_tensor)

        else:
            cosine_similarity_score = cosine_similarity_n(generated_label, real_labels_tensor) 
                
            # Convert generated pseudo labels and true labels to tensors
        generated_label = generated_label.view(-1)
        reward = self.rewarder(feats_x_ulb_w,generated_label)
        reward=reward.view(1)

        generator_loss = self.criterion(reward, torch.ones_like(reward).cuda(self.gpu))
        rewarder_loss = self.criterion(reward, cosine_similarity_score)

        return generator_loss, rewarder_loss