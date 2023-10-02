import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, feature_dim):
        super(Generator, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x

class Rewarder(nn.Module):
    def __init__(self, label_embedding_dim, feature_dim):
        super(Rewarder, self).__init__()

        # Feature processing section
        self.feature_fc = nn.Linear(feature_dim, 128)
        self.feature_norm = nn.LayerNorm(128)

        # Label embedding section
        self.label_embedding = nn.Embedding(100, label_embedding_dim)
        self.label_norm = nn.LayerNorm(label_embedding_dim)

        # Cross-Attention mechanism
        self.cross_attention_fc = nn.Linear(128, 1)

        # MLP (Multi-Layer Perceptron)
        self.mlp_fc1 = nn.Linear(128, 256)
        self.mlp_fc2 = nn.Linear(256, 128)

        # Feed-Forward Network (FFN)
        self.ffn_fc1 = nn.Linear(128, 64)
        self.ffn_fc2 = nn.Linear(64, 1)

    def forward(self, features, label_indices):
        # Process features
        features = self.feature_fc(features)
        features = self.feature_norm(features)

        # Process labels
        label_embed = self.label_embedding(label_indices.to(torch.int64))
        label_embed = self.label_norm(label_embed)

        # Cross-Attention mechanism
        cross_attention_input = torch.cat((features.unsqueeze(0), label_embed.unsqueeze(0)), dim=0)
        cross_attention_weights = torch.softmax(self.cross_attention_fc(cross_attention_input), dim=0)
        cross_attention_output = (cross_attention_weights * cross_attention_input).sum(dim=0)

        # MLP section
        mlp_input = torch.cat((cross_attention_output, label_embed), dim=0)
        mlp_output = F.relu(self.mlp_fc1(mlp_input))
        mlp_output = self.mlp_fc2(mlp_output)

        # FFN section
        ffn_output = F.relu(self.ffn_fc1(mlp_output))
        reward = torch.sigmoid(self.ffn_fc2(ffn_output))
        reward = torch.mean(reward)

        return reward

def cosine_similarity_n(x, y):
    # Calculate cosine similarity
    cosine_similarity = torch.cosine_similarity(x, y)

    # Normalize cosine similarity to the range of 0 to 1
    normalized_similarity = 0.5 * (cosine_similarity + 1)

    return normalized_similarity
