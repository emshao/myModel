import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import autoencoder
import matplotlib.pyplot as plt
from dataloader import fetch_train_data, fetch_test_data

saveImages = None

def trainModel(model, dataname="FashionMNIST", batch=64):

    # Set device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the FashionMNIST dataset and apply transformations
    # train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    # data loaders
    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # random_image, random_label = train_dataset[0]
    # # plt.figure()
    # # plt.imshow(random_image.squeeze().numpy(), cmap='gray')
    # # plt.title(random_label)
    
    # plt.savefig(f'C:\\Users\\Emily Shao\\Desktop\\myModel\\myModel\\images\\originalImage.png')
    # plt.show()

    train_loader = fetch_train_data(dataname, batch)




    # loss and optimize
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(2):

        # what is an epoch? = how many times i should train this model over this dataset
        # is it different than a batch?
        # how many should I have?
        # what iteration?


        for i, (images, labels) in enumerate(train_loader):

            # this is per batch (not too big)

            # how to set epoch, batchsize, learning rate



            images = images.to(device)  # is it modifying the actual numbers within the image? or taking a copy of the dataset
            labels = labels.to(device)

            # set interval to print out image to see the training progress
            # if i%10000==0 :
            #     random_image, random_label = train_dataset[i]
            #     plt.figure()
            #     plt.imshow(random_image.squeeze().numpy(), cmap='gray')
            #     plt.title(random_label)
            #     plt.show()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/10], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")
        


        # write test loop
        # for i in enumerate(test_loader):
            # model.eval()
            # calculate PSNR 
                # input = images
                # output = output
                # use no gradient
            # use PSNR to select best model
            # create a list outside of the different 
        
        
        # save "checkpoint" model from this training loop after every epoch

        random_image = images[0]
        plt.figure()
        plt.imshow(random_image.squeeze().numpy(), cmap='gray')
        plt.title(random_label)
        plt.show()
        plt.savefig(f'C:\\Users\\Emily Shao\\Desktop\\myModel\\myModel\\images\\trainingOut{epoch}.png')
    
    # random_image, random_label = train_dataset[0]
    # plt.figure()
    # plt.imshow(random_image.squeeze().numpy(), cmap='gray')
    # plt.title(random_label)
    # plt.show()

    '''
    save best model
    save from list of PSNR values of different models
    PSNR of each model after every epoch test
    considered within checkpoint to updates best model
    return best model
    '''
    
    return model

    def returnData():
        return saveImages[0]



'''
neural network drop out:

multi-linear perception (NLP)
there's lots of neurons, there will be an overfitting problem

dropout helps solve overfitting problem
dropout probability:
remove using this probability to get rid of some neurons (not connect them)
reduces overfitting

this is a training stragety

dropout is a layer that you are writing in your model

do not test with dropout

model.eval() = automatically skips the drop out function/layer



when you're doing inference with your model,
if you're doing compression and you don't specify model.eval(),
then test.output may have some randomness
'''