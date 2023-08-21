import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, quantizer=None):
        super(Encoder, self).__init__()

        self.down1 = nn.Sequential(
            # in place for transformer
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2), # shape should change/downsampling
            nn.BatchNorm2d(32), # add normalization function here
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(64*3*3, 10)

        self.quantizer = None
        if (quantizer):
            self.quantizer = quantizer
            



    
    def forward(self, x):
        layer1 = self.down1(x)
        layer2 = self.down2(layer1)
        layer3 = self.down3(layer2)

        out = layer3.view(layer3.size(0), -1)  # Flatten the tensor for the fully connected layer
        out = self.fc(out)

        if (self.quantizer):
            out = self.quantizer.quantize(out)

        return out