import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(), "output size", output.size())

        return output


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rand_loader = DataLoader(dataset=RandomDataset(10, 1000), batch_size=2, shuffle=True)
model = SimpleModel(input_size=10, output_size=5).to(device)

if torch.device_count() > 1:
    print("Let's use", torch.device_count(), "GPUs!")
    model = nn.DataParallel(model)
else:
    print("No GPU available")


for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(), "output_size", output.size())
