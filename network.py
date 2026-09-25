import torch
import torch.nn as nn

from modules import position_embedding, FeatureEmbedding, embedding_module_log


class RIRNetwork(nn.Module):
    def __init__(self, L, h, F, device):
        super(RIRNetwork, self).__init__()

        self.device = device

        self.L = L
        self.h = h
        self.F = F
        self.width = 256

        # h is the dim of latent vector
        # 1. Embed Tx and Rx feature vectors to be a (5,h) context
        # self.Tx_embedder =  FeatureEmbedding(h=h, n_occlusion=8)
        # self.Rx_embedder =  FeatureEmbedding(h=h, n_occlusion=8)
        self.Tx_embedder = FeatureEmbedding(N=1024, h=h, n_occlusion=14)
        self.Rx_embedder = FeatureEmbedding(N=1024, h=h, n_occlusion=14)

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
        self.tanh = nn.Tanh()

    def forward(self, tx_loc, rx_loc, tx_feature, rx_feature, t):
        batch_size = tx_loc.shape[0]

        # 1. Get all features embedded

        # print("tx_feature shape: ", tx_feature.shape)
        # print("rx_feature shape: ", rx_feature.shape)

        tx_feature_emb = self.Tx_embedder(tx_feature)  # (batch_size, 5, h)
        rx_feature_emb = self.Rx_embedder(rx_feature)  # (batch_size, 5, h)

        # print("tx_feature_emb shape: ", tx_feature_emb.shape)
        # print("rx_feature_emb shape: ", rx_feature_emb.shape)
        # raise RuntimeError("Debugging shapes")

        batch_size = tx_loc.shape[0]
        # print("tx_loc shape before embedding: ", tx_loc.shape)
        tx_loc_emb = (
            self.loc_encode(tx_loc).permute(1, 0, 2).reshape(batch_size, -1)
        )  # (2 * L + 1, batch_size, 2) -> (batch_size, 2 * L + 1, 2) -> (batch_size, 4 * L + 2)
        # print("tx_loc_emb shape before embedding: ", tx_loc_emb.shape)
        tx_loc_emb = self.location_embed(tx_loc_emb).unsqueeze(1)  # (batch_size, 1, h)

        rx_loc_emb = (
            self.loc_encode(rx_loc).permute(1, 0, 2).reshape(batch_size, -1)
        )  # (2 * L + 1, batch_size, 2) -> (batch_size, 2 * L + 1, 2) -> (batch_size, 4 * L + 2)
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

        # 3. Dot product the t_embed vector with all columns in Tx and Rx
        # print(C.shape, t_embed.shape)
        C = torch.einsum("bij,j->bij", C, t_embed)  # (batch_size, 12, h)
        C_flattened = C.view(C.size(0), -1)  # (batch_size, 12 * h)

        # context = torch.cat([C_flattened, embeddings], dim=1)  # (batch_size, 12 * h + 8)
        context = C_flattened

        x = self.Lrelu(self.fc1(context))  # (15*h, width)

        # x = torch.cat([x, embeddings], dim=1)  # (batch_size, width + 2 * h)
        x = self.Lrelu(self.fc2(x))  # (batch_size, width)

        # x = torch.cat([x, embeddings], dim=1)  # (batch_size, width + 2 * h)
        x = self.Lrelu(self.fc3(x))  # (batch_size, width)

        # x = torch.cat([x, embeddings], dim=1)  # (batch_size, width + 2 * h)
        x = self.Lrelu(self.fc4(x))  # (batch_size, width)

        x = self.fc5(x)

        return x
