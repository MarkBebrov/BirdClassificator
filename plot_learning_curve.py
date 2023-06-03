import torch
import numpy as np
import matplotlib.pyplot as plt

model = torch.load('C:/BIRDFINAL/finetuned_model.pth', map_location=torch.device('cpu'))

train_loss = np.array(model.train_loss)
valid_loss = np.array(model.valid_loss)
train_acc = np.array(model.train_acc)
valid_acc = np.array(model.valid_acc)

epochs = range(1, len(train_loss) + 1)

plt.figure()
plt.plot(epochs, train_loss, 'r', label='Training loss')
plt.plot(epochs, valid_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure()
plt.plot(epochs, train_acc, 'r', label='Training accuracy')
plt.plot(epochs, valid_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
