import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import autoencoder
import matplotlib.pyplot as plt

saveImages = None

def trainModel(model):

    # Set device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the FashionMNIST dataset and apply transformations
    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

    # data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    random_image, random_label = train_dataset[0]
    plt.figure()
    plt.imshow(random_image.squeeze().numpy(), cmap='gray')
    plt.title(random_label)
    plt.show()

    # loss and optimize
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(2):
        for i, (images, labels) in enumerate(train_loader):
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

        random_image = images[0]
        plt.figure()
        plt.imshow(random_image.squeeze().numpy(), cmap='gray')
        plt.title(random_label)
        plt.show()
    
    # random_image, random_label = train_dataset[0]
    # plt.figure()
    # plt.imshow(random_image.squeeze().numpy(), cmap='gray')
    # plt.title(random_label)
    # plt.show()

    
    
    return model

    def returnData():
        return saveImages[0]
