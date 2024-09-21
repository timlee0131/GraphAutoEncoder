import torch.nn as nn

class VanillaGAE(nn.Module):
    def __init__(self, encoder, decoder, loss, n_dim, n_latent) -> None:
        super(VanillaGAE, self).__init__()
        self.encoder = encoder(n_dim, n_latent)
        self.decoder = decoder(n_latent, n_dim)
        self.loss = loss
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z, edge_index)
    
    def loss(self, target, out):
        return self.loss(target, out)
    
    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)