import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self): # make sure to add the de-quantizing step here
        super(Decoder, self).__init__()

        self.down1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.down2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.down3 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(64*3*3, 10)


    """
    cross-scale decoding

    to implement:
    - more complex than simply decoding
    - decoded data need to go to other places

    """
    
    def forward(self, x):
        layer1 = self.down1(x)
        layer2 = self.down2(layer1)
        layer3 = self.down3(layer2)

        out = layer3.view(layer3.size(0), -1)  # Flatten the tensor for the fully connected layer
        out = self.fc(out)

        return out