import numpy as np
np.set_printoptions(precision=3, suppress=True)

# Network architecture
INPUT_SIZE = 2    # Two inputs (A and B)
HIDDEN_SIZE = 4   # Four neurons in hidden layer
OUTPUT_SIZE = 1   # One output (0 or 1)
np.set_printoptions(precision=3, suppress=True)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

weights_input_hidden = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.5
bias_hidden = np.zeros((1, HIDDEN_SIZE))

weights_hidden_output = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.5
bias_output = np.zeros((1, OUTPUT_SIZE))

print("Network initialized with random weights:")
print(f"  Input → Hidden weights shape: {weights_input_hidden.shape}")
print(f"  Hidden → Output weights shape: {weights_hidden_output.shape}")
print(f"\nTotal parameters: {weights_input_hidden.size + bias_hidden.size + weights_hidden_output.size + bias_output.size}")
# 8 + 4 + 4 + 1

def sigmoid(x):
    """Squash values to range (0, 1)"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid: σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def forward_pass(input_data):
    raw_hidden_signal = np.dot(input_data, weights_input_hidden) + bias_hidden
    squashed_hidden_output = sigmoid(raw_hidden_signal)
    
    raw_final_signal = np.dot(squashed_hidden_output, weights_hidden_output) + bias_output
    final_prediction = sigmoid(raw_final_signal)
    
    return raw_hidden_signal, squashed_hidden_output, raw_final_signal, final_prediction

z_h, a_h, z_o, predictions = forward_pass(X)
print(predictions)
print("Forward pass with UNTRAINED network:")
print("-" * 50)
for i in range(len(X)):
    print(f"Input: {X[i]} → Prediction: {predictions[i][0]:.4f} (Target: {y[i][0]})")

print("\n❌ Predictions are garbage — the network hasn't learned anything yet.")

def compute_loss(y_true, y_pred):
    """Mean Squared Error"""
    return np.mean((y_true - y_pred) ** 2)

initial_loss = compute_loss(y, predictions)
print(f"Initial Loss (untrained): {initial_loss:.4f}")
print("\nThis number should decrease as we train.")
print("\n")


# _, _, _, final_predictions = forward_pass(X)

def backward(input_data, expected_output, hidden_layer_without_activ, hidden_layer_activ, output_layer_without_activ, output_layer_activ, learning_rate):
    global weights_input_hidden, bias_hidden, weights_hidden_output, bias_output
    m = input_data.shape[0]

    output_difference = output_layer_activ - expected_output
    output_delta = output_difference * sigmoid_derivative(output_layer_without_activ) # need to understand

    gradient_of_loss_for_hidden_output = np.dot(hidden_layer_activ, output_delta) / m
    grad_bias_output = np.mean(output_delta, axis=0, keepdims=True)


    hidden_error = np.dot(output_delta, weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_without_activ)

    grad_weights_input_hidden = np.dot(X.T, hidden_delta) / m
    grad_bias_hidden = np.mean(hidden_delta, axis=0, keepdims=True)

    weights_hidden_output -= learning_rate * gradient_of_loss_for_hidden_output
    bias_output -= learning_rate * grad_bias_output
    weights_input_hidden -= learning_rate * grad_weights_input_hidden
    bias_hidden -= learning_rate * grad_bias_hidden

print("Backpropagation function defined.")
print("This is the 'learning' part — adjusting weights to reduce error.")

# print(z_o)
learning_rate = 2.0
iterations = 900000

np.random.seed(42)
weights_input_hidden = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.5
bias_hidden = np.zeros((1, HIDDEN_SIZE))
weights_hidden_output = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.5
bias_output = np.zeros((1, OUTPUT_SIZE))

loss_history = []

print("Training started...")
print("-" * 50)

for i in range(iterations):
    # Forward pass
    z_h, a_h, z_o, predictions = forward_pass(X)

    # Calculate loss
    loss = compute_loss(y, predictions)
    loss_history.append(loss)

    # Backward pass (updates weights internally)
    backward(X, y, z_h, a_h, z_o, predictions, learning_rate)

    # Print progress
    if i % 2000 == 0:
        print(f"Iteration {i:5d} | Loss: {loss:.6f}")


_, _, _, final_predictions = forward_pass(X)

print("Final Results After Training:")
print("-" * 50)
print(f"{'Input':<12} {'Target':<10} {'Prediction':<12} {'Rounded':<10}")
print("-" * 50)

for i in range(len(X)):
    pred = final_predictions[i][0]
    rounded = round(pred)
    status = "✅" if rounded == y[i][0] else "❌"
    print(f"{str(X[i]):<12} {y[i][0]:<10} {pred:<12.4f} {rounded:<10} {status}")

print("-" * 50)
print(f"\n🎉 The network learned XOR from random weights!")