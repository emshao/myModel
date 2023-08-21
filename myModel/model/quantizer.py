import torch
import torch.nn as nn

class Quantizer():
    codebook = None

    def __init__(self):
        quantized = False
    
    def quantize(self, x):
        return x
    
    def __init__(self, num_levels):
        self.num_levels = num_levels

    def quantize(self, tensor):
        scale = (tensor.max() - tensor.min()) / (self.num_levels - 1)
        quantized_tensor = ((tensor - tensor.min()) / scale).round() * scale + tensor.min()
        return quantized_tensor
    
#     # use codebook --> vqvae
#     # rounding uses other things



class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e      # number of embeddings
        self.e_dim = e_dim  # embedding size
        self.beta = beta    # hyperparameter

        self.embedding = nn.Embedding(self.n_e, self.e_dim) # codebook, within pytorch
        # this is a matrix (think of matrix)
        # has gradient

        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e) 
        # initialize with uniform distribution
        # shape = number of embeddings and embedding size
        # like a dictionary

    def forward(self, z, test=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        # force encoder to get closer to codebook

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)
        # commitment loss + codebook loss ===> to get input/codebook closer together
        # input is "encoded latent vector representation"
        # purpose of this loss is to train the codebook
        # 

        # preserve gradients
        if not test:
            z_q = z + (z_q - z).detach()
        # because this is discretized
        # z_q gradient point to z
        # use z for smoother calculation
        # this step can only be performed during training, when you are testing don't use this
        # testing you cannot access z
        # when you write the code, write an if (test) statement: 

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
        # loss --> vq / codebook
        # z_q --> output of quantized z (input)
        # min_encodings + indicies ==> the code of the codebook
        # when decompressing, use indicies to refer to embeddings codebook


        # train together, test compress and decompress separately


