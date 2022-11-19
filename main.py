import torch
import wandb
from model import VAE
from torch.utils.tensorboard import SummaryWriter
from utils import load_dataset, load_checkpoint, train, \
    generate_image_from_random_test_image, generate_image_with_random_z
from torchvision.utils import save_image
import os

if __name__ == '__main__':
    # Checkpoint path for saving and loading the model
    ckp_path = 'checkpoints/checkpoint.pt'
    output_dir = './generated_images/'

    # Determine if any GPUs are available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize hyperparameters
    batch_size = 1000
    learning_rate = 1e-3
    num_epochs = 100

    # Initialize wandb
    wandb.init(project="f22-csci780-homework-2")
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size
    }

    # Define SummaryWriter for tensorboard
    writer = SummaryWriter("runs/VAE")

    # Load train and test dataset
    train_loader, test_loader = load_dataset(batch_size)

    # Initialize the model and the Adam optimizer
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    # model.train()
    # train(train_loader, ckp_path, model, optimizer, num_epochs, device, writer)

    # Load the saved model
    model, optimizer, epoch, loss = load_checkpoint(ckp_path, model, optimizer)

    model.eval()
    # Generate image by a random image from the test dataset
    # generate_image_from_random_test_image(test_loader, model, device)
    # Generate image by z with random values

    for i in range(500):
        generated_image = generate_image_with_random_z(model)
        save_image(tensor=generated_image.cpu(), fp=os.path.join(output_dir, '{}_image.jpg'.format(i)), nrow=256,
                   pad_value=1)




