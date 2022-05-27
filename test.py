import sys
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from network import generator

args = sys.argv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_set = datasets.MNIST("mnist/", train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

print(args[1])
G = generator()
G.load_state_dict(torch.load(args[1]))
G = G.to(device)
G.eval()

for i in train_loader:
    print("real")
    plt.title("real")
    plt.imshow(i[0][0].reshape(28, 28))
    plt.savefig("real.png")
    real_inputs = i[0][0]
    noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
    noise = noise.to(device)
    fake_inputs = G(noise)
    print("fake")
    plt.title("fake")
    plt.imshow(fake_inputs[0][0].cpu().detach().numpy().reshape(28, 28))
    plt.savefig("fake.png")
    break
