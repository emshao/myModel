import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model.encoder import Encoder
from model.quantizer import Quantizer
from model.decoder import Decoder
from dataloader import fetch_train_data

class AutoEncoder(nn.Module):
    def __init__(self, channel, kernal_sz, strd, pdd):
        super().__init__()

        self.quantizer = Quantizer(512, channel, 0) # numbers??

        self.encoder = Encoder(channel, kernal_sz, strd, pdd, self.quantizer)
        codebook = self.quantizer.return_codebook()
        self.decoder = Decoder(channel, kernal_sz, strd, pdd, codebook)

        self.loss = nn.CrossEntropyLoss() # how to implement?????
    
   
    def forward(self, input):
        '''
        input size = torch.Size([batch, channels, height, width])
        label size = torch.Size([batch])

        remember forward is called iteratively on the batches, model is updated with generalized changes after each batch 

        '''

        encoded = self.encoder(input)
        decoded = self.decoder(encoded)

        return decoded





    def createModel(self):
        model = AutoEncoder()
        return model
    




if __name__ == "__main__":
    # obj = AutoEncoder()
    
    # check size of each batch and image
    train_loader = fetch_train_data("FashionMNIST", 128)    # torch.Size([128, 1, 28, 28])
    train_loader2 = fetch_train_data("CIFAR-10", 128)       # torch.Size([128, 3, 32, 32])

    for i, (images, labels) in enumerate(train_loader2):
        print("Batch:", i)
        print(images.shape)
        print(labels.shape)
        break