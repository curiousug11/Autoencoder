import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
}


def mlp(sizes, activation):
    """Linear layers with `activation` between them; no activation after the last one."""
    layers = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n_in, n_out))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Symmetric fully connected autoencoder, e.g. 52-128-64-32-16-8-16-32-64-128-52.

    The bottleneck and the output are linear: z-normalised spectra are unbounded.
    """

    def __init__(self, n_in=52, hidden=(128, 64, 32, 16), latent=8, activation=nn.GELU):
        super().__init__()
        enc_sizes = [n_in, *hidden, latent]
        self.encoder = mlp(enc_sizes, activation)
        self.decoder = mlp(enc_sizes[::-1], activation)

    def forward(self, x):
        return self.decoder(self.encoder(x))
