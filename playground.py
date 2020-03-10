import argparse
from torchvision import datasets, transforms
import torch
from net import AlexNetPlusLatent

parser = argparse.ArgumentParser()
parser.add_argument('--bits', type=int, default=48)
parser.add_argument('--pretrained', type=float, default=91.33)
args = parser.parse_args()

def load_data():
    transform_test = transforms.Compose(
        [transforms.Resize(227),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    testset = datasets.CIFAR10(root='./data', train=False, download=True,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset)

    return testloader

def binary_output(dataloader):
    net = AlexNetPlusLatent(args.bits)
    net.load_state_dict(torch.load('./model/{}'.format(args.pretrained)))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use device: ' + str(device))
    net.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs, _ = net(inputs)
            print(batch_idx, outputs.data[0])

if __name__ == "__main__":
    _loader = load_data()
    binary_output(_loader)