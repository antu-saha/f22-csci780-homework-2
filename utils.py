import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from random import random
import matplotlib.pyplot as plt
import numpy as np
import random
import wandb


def load_dataset(batch_size):
    """
    Create dataloaders to feed data into the neural network
    Default MNIST dataset is used and standard train/test split is performed
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
        batch_size=1)
    return train_loader, test_loader


def save_checkpoint(ckp_path, model, epoch, optimizer, loss):
    """
    This function will save the checkpoint at the checkpoint path.
    :param ckp_path: checkpoint path,
    :param model: model,
    :param epoch: epoch,
    :param optimizer: optimizer,
    :param loss: loss,
    :return: nothing.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, ckp_path)


def load_checkpoint(ckp_path, model, optimizer):
    """
    This function will load the checkpoint.
    :param ckp_path: checkpoint path,
    :param model: model,
    :param optimizer: optimizer,
    :return: model, optimizer, epoch, loss.
    """
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train(train_loader, ckp_path, model, optimizer, num_epochs, device, writer):
    """
    This function is for training the network for a given number of epochs.
    The loss after every epoch is printed.
    :param train_loader: train dataset,
    :param ckp_path: checkpoint path,
    :param model: model,
    :param optimizer: optimizer,
    :param num_epochs: total number of epochs,
    :param device: device,
    :param writer: writer for tensorboard,
    :return: nothing.
    """
    print(f'Entering into the training loop.....')
    wandb.watch(model)
    n_total_steps = len(train_loader)
    running_loss = 0.0

    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader, 0):
            images, _ = data
            images = images.to(device)

            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = model(images)

            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, images, reduction='sum') + kl_divergence

            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (idx + 1) % 5 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}, Step [{idx + 1}/{n_total_steps}]'
                      f'Loss: {loss.item():.4f}]')
                # writer.add_graph(model, images)
                # writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + idx)
                wandb.log({"Epoch": epoch + 1, "Loss": loss})
                running_loss = 0.0

        print('\nEpoch {}: Loss {}'.format(epoch + 1, loss))
        save_checkpoint(ckp_path, model, epoch, optimizer, loss)
        if (epoch + 1) % 10 == 0:
            print(f'Checkpoint saved after epoch no. {epoch + 1}')
        print('\n\n')

    save_checkpoint(ckp_path, model, epoch, optimizer, loss)
    print(f'Trained model saved.')


def generate_image_from_random_test_image(test_loader, model, device):
    """
    This function will select an image from the test dataset,
    feed that image to the trained model, and generate a similar image.
    :param test_loader: the test dataset,
    :param model: the trained model,
    :param device: the selected device.
    """
    with torch.no_grad():
        for data in random.sample(list(test_loader), 1):
            # Select a random image from the test set
            image, label = data
            image = image.to(device)
            img = np.transpose(image[0].cpu().numpy(), [1, 2, 0])
            # Plot the random image
            plt.subplot(121)
            plt.imshow(np.squeeze(img))
            plt.show()

            # Pass the image through the trained model
            out, mu, logVAR = model(image)
            # Generate image
            generated_image = np.transpose(out[0].cpu().numpy(), [1, 2, 0])
            # Plot the generated image
            plt.subplot(122)
            plt.imshow(np.squeeze(generated_image))
            plt.show()
            break


def generate_image_with_random_z(model):
    """
    This function will initialize z with truncated normal distribution,
    and generate images with z.
    :param model: trained model,
    :return: generated image.
    """
    with torch.no_grad():
        # z = torch.rand(1, 256)
        # print(z)
        z = torch.empty(1, 256)
        torch.nn.init.trunc_normal_(z, 0, 1, -100, 100)
        # print(z)
        generated_image = torch.sigmoid(model.decoder(z))
        # print(f'Size of the output image: {generated_image.shape}')
        plt.subplot(122)
        generated_image = np.squeeze(generated_image)
        plt.imshow(generated_image)
        # plt.show()
    return generated_image
