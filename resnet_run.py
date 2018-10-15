import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from resnet import resnet18, resnet34, resnet50, resnet101
import sys
import matplotlib.pyplot as plt
import numpy as np

def train(train_dataset, val_dataset, device, learning_rate):
    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size, 
                                              shuffle=False)

    # model = resnet18(pretrained=True)
    # model = resnet34(0)
    model = resnet18(0)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    curr_lr = learning_rate

    validation_accuracy_log = []
    validation_loss_log = []
    loss_log = []
    for epoch in range(num_epochs):
           
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 10 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            if (i+1) % 40 == 0:
                avg_loss = running_loss/40
                loss_log.append(avg_loss)
                running_loss = 0
                val_accuracy, val_loss = test_validation(model, device, val_loader, criterion)
                validation_accuracy_log.append(val_accuracy)
                validation_loss_log.append(val_loss)
                

        if (epoch+1) % 3 == 0:
            plot_results(total_step, epoch, loss_log, validation_loss_log, validation_accuracy_log)
            plt.show()

        # Decay learning rate
        if (epoch+1) % 10 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

    return model

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test_validation(model, device, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        count = 0
        running_loss = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            count += 1

        print('Accuracy of the model on the validation images: {} %'.format(100 * correct / total))
        print('Loss of the model on the validation images: {}'.format(running_loss / count))
    accuracy = correct / total
    loss = running_loss / count
    return accuracy, loss


def plot_results(total_step, epoch, loss_log, validation_loss_log, validation_accuracy_log):
    iteration_loss_log = np.linspace(0,(epoch+1)*total_step,len(loss_log))
    iteration_accuracy_log = np.linspace(0,(epoch+1)*total_step,len(validation_accuracy_log))
    print('iteration_loss_log')
    print(iteration_loss_log)
    print('iteration_accuracy_log')
    print(iteration_accuracy_log)
    # plt.plot(iteration_log, loss_log)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(iteration_loss_log, loss_log, 'r--', label = 'training loss')
    plt.plot(iteration_loss_log, validation_loss_log, 'b-.', label = 'validation loss')
    plt.title('Loss for validation and testing')
    plt.xlabel('iterations')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.subplot(212)
    plt.plot(iteration_accuracy_log, validation_accuracy_log, 'r--', label = 'training accuracy')
    plt.title('Accuracy for validation')
    plt.xlabel('iterations')
    legend = plt.legend(loc='lower right', shadow=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)


## MAIN ##

cuda_check = torch.cuda.is_available()
if not cuda_check:
    print('No cuda device found. Training will be way faster with cuda')
    sys.exit()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 10
learning_rate = 0.00001
# learning_rate = 0.001
batch_size = 30
# batch_size = 10


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

# Image preprocessing modules
transform = transforms.Compose([
    # transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(600),
    # transforms.Resize(600),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # normalize
    ])


train_dataset = ImageFolder('./dataset/train', transform=transform)

val_dataset = ImageFolder('./dataset/val', transform=transform)


#This code just visualises some of the images from the loader 
# #-------------------------------------------------------
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                                batch_size=batch_size, 
#                                                shuffle=True)

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))

# # get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))

# plt.show()

# sys.exit()

# #-------------------------------------------------------


model = train(train_dataset, val_dataset, device, learning_rate)

# Test the model
val_accuracy, val_loss = test_validation(model, device, val_loader, criterion)

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')