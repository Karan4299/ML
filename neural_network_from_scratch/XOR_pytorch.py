import torch
import torch.nn as nn
import torch.optim as optim

X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [
    [0],
    [1],
    [1],
    [0]
]

X_tensor = torch.FloatTensor(X)
Y_tensor = torch.FloatTensor(y)

class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.hidden2 = nn.Linear(4, 4)
        self.hidden3 = nn.Linear(4, 8)
        self.hidden4 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, input_data):
        hidden_layer_output = self.relu(self.hidden(input_data))
        hidden_layer_output2 = self.relu(self.hidden2(hidden_layer_output))
        hidden_layer_output3 = self.relu(self.hidden3(hidden_layer_output2))
        hidden_layer_output4 = self.relu(self.hidden4(hidden_layer_output3))
        out_layer_output = self.sigmoid(self.output(hidden_layer_output4))

        return out_layer_output

torch.manual_seed(42)
model = XORNet()

MSE_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=2.0)

pytorch_loss_history = []

for i in range(10000):
    # Forward pass
    prediction = model(X_tensor)

    # calc mean squares error
    loss = MSE_fn(prediction, Y_tensor)

    # optional: ffeed in history
    pytorch_loss_history.append(loss.item())

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 2000 == 0:
        print(f"Iteration {i:5d} | Loss: {loss.item():.6f}")

print("-" * 50)
print(f"Iteration {10000:5d} | Loss: {pytorch_loss_history[-1]:.6f}")

print("\nPyTorch Final Predictions:")
print("-" * 50)
with torch.no_grad():
    final_preds = model(X_tensor)
    for i in range(len(X)):
        pred = final_preds[i].item()
        print(f"Input: {X[i]} → Prediction: {pred:.4f} (Target: {y[i][0]})")