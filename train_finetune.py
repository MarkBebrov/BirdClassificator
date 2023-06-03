import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from load_data import load_data
from model import build_model
from train import train_model
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    # Загрузите предобученную модель
    model = torch.load('C:/BIRDFINAL/trained_model.pth')

    # Измените последний слой модели
    num_classes = 525  # Замените на количество классов в вашем датасете
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Получите данные для обучения
    dataloaders, dataset_sizes, _ = load_data()

    # Определите функцию потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Выполните дообучение модели
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=30)

    # Сохраните дообученную модель
    torch.save(model, 'C:/BIRDFINAL/finetuned_model.pth')

    # Получите статистику обучения и валидации
    train_loss = model.train_loss
    valid_loss = model.valid_loss
    train_acc = model.train_acc
    valid_acc = model.valid_acc

    # Постройте графики потерь и точности
    epochs = range(1, len(train_loss) + 1)

    # График потерь
    plt.figure()
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.plot(epochs, valid_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # График точности
    plt.figure()
    plt.plot(epochs, train_acc, 'r', label='Training accuracy')
    plt.plot(epochs, valid_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
