import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import autoencoder
import matplotlib.pyplot as plt
from dataloader import fetch_test_data

def testModel(model, data_name):
    # Set device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = fetch_test_data(data_name)


    # random_image, random_label = test_dataset[0]
    # plt.figure()
    # plt.imshow(random_image.squeeze().numpy(), cmap='gray')
    # plt.title(random_label)
    # plt.show()


    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Accuracy of the model on the test images: {(100 * correct / total):.2f}%")

    
    # with torch.no_grad():
    #     rand_output = model(torch.tensor([random_image, 1]))

    #     plt.figure(1)
    #     plt.imshow(random_image.squeeze().numpy(), cmap='gray')
    #     plt.title(random_label)

    #     plt.figure(2)
    #     plt.imshow(rand_output.squeeze().numpy(), cmap='gray')
    #     plt.title("reconstructed")
    #     plt.show()


