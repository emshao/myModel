import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model.encoder import Encoder
from model.quantizer import Quantizer
from model.decoder import Decoder

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.quantizer = Quantizer(8)

        # input
        # torch.Size([64, 1, 28, 28])
        # torch.Size([64])

        # self.encoder = Encoder(quantizer=self.quantizer)
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.loss = nn.CrossEntropyLoss()

        


    
    def forward(self, input):
        '''
        image: torch.Size([64, 1, 28, 28])
        labels: torch.Size([64])
        
        '''

        encoded = self.encoder(input)
        # print(encoded.shape)


        # decoded = self.decoder(encoded)

        return encoded
        # return decoded

    def createModel(self):
        model = AutoEncoder()
        return model
    


# if __name__ == "__main__":
#     obj = AutoEncoder()
#     # Load the FashionMNIST dataset and apply transformations
#     train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

#     # data loaders
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


#     for i, (images, labels) in enumerate(train_loader):
#         print(images.shape)
#         print(labels.shape)
#         break