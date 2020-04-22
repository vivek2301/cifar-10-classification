import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from model import CNN
import matplotlib.pyplot as plt

N_EPOCHS = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("1. Transform data to zero mean and one standard deviation.\n")
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    classify(transform_train, transform_test)

    print("2. Images with pixel value between zero and one.\n")
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    classify(transform_train, transform_test)

def classify(transform_train, transform_test):

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)                                    
    trainset, validset = torch.utils.data.random_split(trainset, [45000, 5000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                shuffle=True, num_workers=4)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64,
                                                shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=4)

    model = CNN()

    #Adam optimizer and Cross entropy Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    #Transfer model and criterion to GPU
    model = model.to(device)
    criterion = criterion.to(device)

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    best_valid_loss = float('inf')
    PATH = './cifar_net.pth'

    for epoch in range(N_EPOCHS):  # loop over the dataset multiple times

        print("EPOCH: %d" % (epoch+1))
        train_loss, train_acc = train(model, trainloader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, validloader, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), PATH)

        train_loss_list.append(train_loss)
        train_acc_list.append(100 * train_acc)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(100 * valid_acc)

    #Plot loss and accuracy
    plotLoss(train_loss_list, valid_loss_list)
    plotAccuracy(train_acc_list, valid_acc_list)

    model = CNN()
    model.load_state_dict(torch.load(PATH))

    #Transfer model to GPU
    model = model.to(device)

    test_loss, test_acc = evaluate(model, testloader, criterion)
    print('Loss of the network on test images: %f ' % (test_loss))
    print('Accuracy of the network on test images: %d %%' % (100 * test_acc))

#Function to train the model
def train(model, dataloader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    running_loss = 0.0
    running_acc = 0.0
    model.train()

    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc

        # print statistics
        running_loss += loss.item()
        running_acc += acc
        if i % 200 == 199:    # print every 2000 mini-batches
            print('loss: %.3f accuracy: %.3f' %
                  (running_loss / 200, running_acc / 200))
            running_loss = 0.0
            running_acc = 0.0
        
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)


#Forward pass for classification
def evaluate(model, dataloader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def accuracy(outputs, labels):
    #round predictions to the closest integer
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    acc = correct / total
    return acc

#Plot training and validation loss
def plotLoss(train_loss, valid_loss):
    epochs = range(N_EPOCHS)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, valid_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

#Plot training and validation Accuracy
def plotAccuracy(train_acc, valid_acc):
    epochs = range(N_EPOCHS)
    plt.plot(epochs, train_acc, 'g', label='Training Accuracy')
    plt.plot(epochs, valid_acc, 'b', label='validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
