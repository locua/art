import argparse
import os
import random
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils


# Define the generator network
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


# Define the training function
def train(args, dataloader, generator, discriminator, optimizer_G, optimizer_D, device):
    for epoch in range(args.num_epochs):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            # Generate fake images and load reals
            real_images = data[0].to(device)
            noise = torch.randn(real_images.size(0), args.nz, 1, 1, device=device)
            fake_images = generator(noise).detach()

            # Check for tensor size mismatches and adjust the tensors accordingly
            # if real_images.shape != fake_images.shape:
            #     new_size = min(real_images.shape[2], fake_images.shape[2])
            #     real_images = F.interpolate(real_images, size=new_size)
            #     fake_images = F.interpolate(fake_images, size=new_size)

            # Resize images to fixed size
            real_images = F.interpolate(real_images, size=args.image_size, mode='bilinear', align_corners=False)
            fake_images = F.interpolate(fake_images, size=args.image_size, mode='bilinear', align_corners=False)


            # Train the discriminator with real images
            discriminator.zero_grad()
            output_real = discriminator(real_images)

            # train the discriminator with them
            output_fake = discriminator(fake_images)
            loss_D = -torch.mean(torch.log(output_real) + torch.log(1 - output_fake))
            loss_D.backward()
            optimizer_D.step()

            # Train the generator to fool the discriminator
            generator.zero_grad()
            output_fake = discriminator(fake_images)
            loss_G = -torch.mean(torch.log(output_fake))
            loss_G.backward()
            optimizer_G.step()

            if i % args.print_freq == 0:
                print(f"Epoch {epoch}, Batch {i}: D_loss = {loss_D.item()}, G_loss = {loss_G.item()}")

            if i % args.save_freq == 0:

                # Save a batch of generated images
                #vutils.save_image(fake_images, os.path.join(args.output_dir, f"epoch_{epoch}_batch_{i}.png"))

                # Generate a single fake image and save it to a file
                with torch.no_grad():
                    noise = torch.randn(1, args.nz, 1, 1, device=device)
                    fake_image = generator(noise).squeeze()
                    fake_image = (fake_image + 1) / 2  # Unnormalize the image
                    vutils.save_image(fake_image, os.path.join(args.output_dir, f"epoch_{epoch}_batch_{i}.png"))

        # Save the generator model after each epoch
        torch.save(generator.state_dict(), os.path.join(args.output_dir, f"generator_epoch_{epoch}.pt"))
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f"discriminator_epoch_{epoch}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a GAN on a folder of images')
    parser.add_argument('--dataroot', type=str, required=True, help='Path to folder of images')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=64, help='Size of images (will be resized)')
    parser.add_argument('--nz', type=int, default=100, help='Size of generator input noise vector')
    parser.add_argument('--ngf', type=int, default=64, help='Size of generator feature maps')
    parser.add_argument('--ndf', type=int, default=64, help='Size of discriminator feature maps')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam optimizer beta1')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency (in batches)')
    parser.add_argument('--save_freq', type=int, default=100, help='Save frequency (in batches)')
    args = parser.parse_args()

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from torch.utils.data.dataloader import default_collate

    # Set up data loader
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.ImageFolder(root=args.dataroot, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=default_collate)


    # Set up generator and discriminator
    generator = Generator(args.nz, args.ngf, 3)
    discriminator = Discriminator(args.ndf, 3)
    generator.to(device)
    discriminator.to(device)

    # Set up optimizer
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Train the GAN
    train(args, dataloader, generator, discriminator, optimizer_G, optimizer_D, device)

           

