import torch
import torch.nn as nn

from modules import FeatureEmbedding, embedding_module_log


class RIRNetwork(nn.Module):
    def __init__(self, L, h, F, device=None, n_rays=1024, n_occlusion=14):
        super(RIRNetwork, self).__init__()

        self.L = L
        self.h = h
        self.F = F

        # h is the dim of latent vector
        # 1. Embed Tx and Rx feature vectors to be a (5,h) context
        self.Tx_embedder = FeatureEmbedding(
            N=n_rays, h=h, n_occlusion=n_occlusion
        )
        self.Rx_embedder = FeatureEmbedding(
            N=n_rays, h=h, n_occlusion=n_occlusion
        )

        # 2. Set time embedder
        self.time_encode = embedding_module_log(num_freqs=self.L)
        self.loc_encode = embedding_module_log(num_freqs=self.L, max_freq=7)

        self.time_transform = nn.Sequential(
            nn.Linear(2 * self.L + 1, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(64, h),
        )

        self.location_embed = nn.Sequential(
            nn.Linear(6 * self.L + 3, h), nn.LeakyReLU(negative_slope=0.1)
        )

        # 3. Learnable embeddings
        for k in range(4):
            self.register_parameter(
                "orient_{}".format(k),
                nn.Parameter(torch.randn(64) / (64**0.5), requires_grad=True),
            )

        for k in range(2):
            self.register_parameter(
                "channel_{}".format(k),
                nn.Parameter(torch.randn(64) / (64**0.5), requires_grad=True),
            )

        # 4. MLP network with skip connection

        self.fc1 = nn.Linear(12 * h, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 512)
        self.fc5 = nn.Linear(512, F)

        self.Lrelu = nn.LeakyReLU(negative_slope=0.1)
    def forward(self, tx_loc, rx_loc, tx_feature, rx_feature, t):
        batch_size = tx_loc.shape[0]

        tx_feature_emb = self.Tx_embedder(tx_feature)  # (batch_size, 5, h)
        rx_feature_emb = self.Rx_embedder(rx_feature)  # (batch_size, 5, h)

        tx_loc_emb = (
            self.loc_encode(tx_loc).permute(1, 0, 2).reshape(batch_size, -1)
        )
        tx_loc_emb = self.location_embed(tx_loc_emb).unsqueeze(1)  # (batch_size, 1, h)

        rx_loc_emb = (
            self.loc_encode(rx_loc).permute(1, 0, 2).reshape(batch_size, -1)
        )
        rx_loc_emb = self.location_embed(rx_loc_emb).unsqueeze(1)  # (batch_size, 1, h)

        Tx_embed = torch.concatenate(
            [tx_feature_emb, tx_loc_emb], dim=1
        )  # (batch_size, 6, h)
        Rx_embed = torch.concatenate(
            [rx_feature_emb, rx_loc_emb], dim=1
        )  # (batch_size, 6, h)

        # Concatenate them to form the whole context
        C = torch.cat([Tx_embed, Rx_embed], dim=1)  # (batch_size, 12, h)

        # 2. Get time embedded
        t_embed = self.time_encode(t).float()  # (2*L + 1,)
        t_embed = self.time_transform(t_embed)  # (batch_size, h)

        C = torch.einsum("bij,j->bij", C, t_embed)  # (batch_size, 12, h)
        C_flattened = C.view(C.size(0), -1)  # (batch_size, 12 * h)

        context = C_flattened

        x = self.Lrelu(self.fc1(context))  # (15*h, width)
        x = self.Lrelu(self.fc2(x))  # (batch_size, width)
        x = self.Lrelu(self.fc3(x))  # (batch_size, width)
        x = self.Lrelu(self.fc4(x))  # (batch_size, width)

        x = self.fc5(x)

        return x
