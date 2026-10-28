import torch
import torch.nn as nn

input_data = torch.tensor([
    [1.0, 6.0],
    [2.0, 8.0],
    [3.0, 4.0],
    [4.0, 7.0],
    [3.0, 2.0]
], dtype=torch.float32)

expected_output = torch.tensor([
    [1],
    [1],
    [0],
    [1],
    [0]
], dtype=torch.float32)

test_input = torch.tensor([
    [10.0, 6.0],
], dtype=torch.float32)

class HiringPredictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(in_features=input_data.shape[1], out_features=4)
        self.layer2 = nn.Linear(in_features=4, out_features=8)
        self.layer3 = nn.Linear(in_features=8, out_features=4)
        self.layer4 = nn.Linear(in_features=4, out_features=1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = torch.relu(x)

        x = self.layer3(x)
        x = torch.relu(x)

        x = self.layer4(x)
        x = torch.sigmoid(x)

        return x

model = HiringPredictor()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for itr in range(10000):
    # forward pass
    model_output = model.forward(input_data)
    
    # calc loss
    loss_in_step = loss_fn(model_output, expected_output)

    # clear optimizer
    optimizer.zero_grad()

    # calck gradient
    loss_in_step.backward()

    # apply the loss / gradient weight set
    optimizer.step()


model_output = model(test_input)
print(model_output)

