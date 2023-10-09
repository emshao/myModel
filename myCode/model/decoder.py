import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, channel_in, channel_out, kernal_sz, strd, pdd, codebook): 
        
        # make sure to add the de-quantizing step here
        # to make model more abstract, add parameters for in_size, out_size

        super(Decoder, self).__init__()

        self.down1 = nn.Sequential(
            nn.ConvTranspose2d(channel_in, 64, kernel_size=kernal_sz, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.down2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=kernal_sz, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.down3 = nn.Sequential(
            nn.ConvTranspose2d(32, channel_out, kernel_size=kernal_sz, stride=2, padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU())
            # nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(625, 10)

        # compare accuracies



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

        # print(layer3.shape)

        out = layer3.view(layer3.size(0), -1)  # Flatten the tensor for the fully connected layer
        # print(out.size)

        out = self.fc(out)

        return out