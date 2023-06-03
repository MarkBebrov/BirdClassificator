import torch
from train import train_model
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from load_data import load_data 
from model import build_model
import matplotlib.pyplot as plt


def plot_loss_accuracy(train_loss, valid_loss, train_acc, valid_acc):
    epochs = range(1, len(train_loss) + 1)

    # Plot loss
    plt.figure()
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.plot(epochs, valid_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, train_acc, 'r', label='Training accuracy')
    plt.plot(epochs, valid_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    # Проверка доступности видеокарты
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # define the number of classes
    num_classes = 525  # replace with the number of bird classes in your dataset

    # get the data
    dataloaders, dataset_sizes, _ = load_data() 

    # build the model
    model = build_model(num_classes)

    # define the loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # train the model
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=10)

    # save the trained model
    torch.save(model, 'trained_model.pth')  # Save entire model, not just the state_dict

    # get the training and validation statistics
    train_loss = model.train_loss
    valid_loss = model.valid_loss
    train_acc = model.train_acc
    valid_acc = model.valid_acc

    # plot the loss and accuracy
    plot_loss_accuracy(train_loss, valid_loss, train_acc, valid_acc)
