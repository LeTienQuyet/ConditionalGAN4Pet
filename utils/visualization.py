import os

import matplotlib.pyplot as plt

def plot_loss(num_epochs, lossesGen, lossesDis, save_pth):
    epochs = range(1, num_epochs+1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, lossesGen, label='Generator Loss')
    plt.plot(epochs, lossesDis, label='Discriminator Loss')
    plt.title('Generator & Discriminator loss arcording to Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_pth, "loss.png"), dpi=300, bbox_inches='tight')