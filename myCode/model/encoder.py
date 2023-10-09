import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, channel, kernal_sz, strd, pdd, quantizer=None):
        super(Encoder, self).__init__()

        self.down1 = nn.Sequential(
            # in place for transformer
            nn.Conv2d(channel, 32, kernel_size=kernal_sz, stride=strd, padding=pdd),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kernal_sz, stride=strd, padding=pdd),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=kernal_sz, stride=strd, padding=pdd),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.lin = nn.Linear(1024, 64)

        self.quantizer = None
        if (quantizer):
            self.quantizer = quantizer
            

    
    def forward(self, x):

        # how to periodically print a picture?
        layer1 = self.down1(x)
        layer2 = self.down2(layer1)
        layer3 = self.down3(layer2)

        print(layer3.shape)

        if (self.quantizer):
            loss, z_q, perplexity, min_encodings, min_encoding_indices = self.quantizer.compress(layer3)

        return layer3


        return z_q, loss
    
    def return_codebook(self):
        return self.quantizer.get_codebook()