import torch
import torch.nn as nn
import numpy as np


def position_embedding(pos, L):
    """
    Generate a position embedding for a single integer value.

    Args:
    pos (int): The position to be encoded.
    L (int): The length of the embedding (half the final vector length).

    Returns:
    np.ndarray: The position embedding vector of shape (2L,).
    """
    assert L > 0, "L must be a positive integer."

    embedding = torch.zeros((2 * L,))

    for i in range(L):
        temp = torch.pow(torch.tensor(10000.0), 2 * i / L)
        embedding[2 * i] = torch.sin(pos / temp)
        embedding[2 * i + 1] = torch.cos(pos / temp)

    return embedding


class FeatureEmbedding(nn.Module):
    """
    CNN-based encoder that produces the same (B, 5, h) token tensor as the
    old MLP implementation, but its weights are independent of the ray count N.
    ─────────────────────────────────────────────────────────────────────────
    Expected flattened feature layout per point: [6*N | n_occlusion]
      0 …   N-1           : distances
      N … 2N-1           : mean-distances
     2N … 3N-1           : variance-distances
     3N … 6N-1           : normals  (x,y,z interleaved)
     6N ... end           : occlusion/global histogram or flags
    """

    def __init__(
        self,
        N: int,
        h: int = 256,
        n_occlusion: int = 8,
        k: int = 3,  # convolution kernel
        mid: int = 64,
    ):  # hidden channels inside each branch
        super().__init__()
        self.h = h
        self.n_occlusion = n_occlusion
        self.k = k
        self.N = N

        def _branch(c_in: int) -> nn.Sequential:
            return nn.Sequential(
                # nn.Conv1d(c_in,   mid, k, padding=k // 2),
                # nn.LeakyReLU(0.1, inplace=True),
                # nn.Conv1d(mid,    h,   k, padding=k // 2),
                # nn.LeakyReLU(0.1, inplace=True),
                nn.Conv1d(c_in, h, k, padding=k // 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.AdaptiveAvgPool1d(1),  # -> (B, h, 1)
            )

        # four geometric branches (dist / mean / var / normals)
        self.dist_enc = _branch(1)
        self.mean_dist_enc = _branch(1)
        self.var_dist_enc = _branch(1)
        self.normal_enc = _branch(3)  # normals ­(x,y,z)

        # occlusion → small MLP to size h
        self.occ_enc = nn.Sequential(
            nn.Linear(n_occlusion, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, h),
        )

    # ------------------------------------------------------------------ #
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        feat : (B, 6·N + 8)  flattened feature vector

        Returns
        -------
        tokens : (B, 5, h)   order = [dist, mean, var, normals, occ]
        """
        B, D = feat.shape
        N = self.N
        assert D == 6 * N + self.n_occlusion, (
            f"Expected feature vector of shape (B, {6 * N + self.n_occlusion}), "
            f"got {feat.shape} instead."
        )

        distances = feat[:, :N]  # (B, N)
        mean_distances = feat[:, N : 2 * N]
        var_distances = feat[:, 2 * N : 3 * N]
        normals_flat = feat[:, 3 * N : 6 * N]  # (B, 3N)
        occlusion = feat[:, 6 * N :]

        normals = normals_flat.view(B, N, 3).permute(0, 2, 1)  # (B, 3, N)

        dist_map = distances.unsqueeze(1)
        mean_map = mean_distances.unsqueeze(1)
        var_map = var_distances.unsqueeze(1)

        dist_tok = self.dist_enc(dist_map).squeeze(-1)  # (B, h)
        mean_tok = self.mean_dist_enc(mean_map).squeeze(-1)
        var_tok = self.var_dist_enc(var_map).squeeze(-1)
        norm_tok = self.normal_enc(normals).squeeze(-1)  # (B, h)

        occ_tok = self.occ_enc(occlusion)  # (B, h)

        tokens = torch.stack(
            [dist_tok, mean_tok, var_tok, norm_tok, occ_tok], dim=1
        )  # (B, 5, h)
        return tokens


class embedding_module_log(nn.Module):
    def __init__(
        self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, include_in=True
    ):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(
            2.0
            ** torch.from_numpy(np.linspace(start=0.0, stop=max_freq, num=num_freqs)),
            requires_grad=False,
        )
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input * freq))
        return torch.stack(out_list)
