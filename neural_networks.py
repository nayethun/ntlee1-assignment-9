import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
        self.cache = {}
        self.grads = {}
    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        Z1 = np.dot(X, self.W1) + self.b1  # Linear activation
        # Apply activation function
        if self.activation_fn == 'tanh':
            A1 = np.tanh(Z1)
        elif self.activation_fn == 'relu':
            A1 = np.maximum(0, Z1)
        elif self.activation_fn == 'sigmoid':
            A1 = 1 / (1 + np.exp(-Z1))
        else:
            raise ValueError("Unsupported activation function")
        Z2 = np.dot(A1, self.W2) + self.b2
        # Output activation (sigmoid for binary classification)
        A2 = 1 / (1 + np.exp(-Z2))
        out = A2
        # TODO: store activations for visualization
        self.cache['X'] = X
        self.cache['Z1'] = Z1
        self.cache['A1'] = A1
        self.cache['Z2'] = Z2
        self.cache['A2'] = A2
        return out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        m = y.shape[0]
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        Z1 = self.cache['Z1']

        # Compute dZ2
        dZ2 = A2 - y  # derivative of loss w.r.t Z2

        # Compute gradients for W2 and b2
        dW2 = (1 / m) * np.dot(A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Backpropagate to hidden layer
        dA1 = np.dot(dZ2, self.W2.T)

        # Compute derivative of activation function
        if self.activation_fn == 'tanh':
            dZ1 = dA1 * (1 - np.power(np.tanh(Z1), 2))
        elif self.activation_fn == 'relu':
            dZ1 = dA1 * (Z1 > 0).astype(float)
        elif self.activation_fn == 'sigmoid':
            sig_Z1 = 1 / (1 + np.exp(-Z1))
            dZ1 = dA1 * sig_Z1 * (1 - sig_Z1)
        else:
            raise ValueError("Unsupported activation function")

        # Compute gradients for W1 and b1
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # TODO: update weights with gradient descent
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        # TODO: store gradients for visualization
        self.grads['dW2'] = dW2
        self.grads['db2'] = db2
        self.grads['dW1'] = dW1
        self.grads['db1'] = db1

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.cache['A1']
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap='bwr',
        alpha=0.7,
    )
    ax_hidden.set_title('Hidden Layer Feature Space')
    ax_hidden.set_xlabel('Neuron 1')
    ax_hidden.set_ylabel('Neuron 2')
    ax_hidden.set_zlabel('Neuron 3')

    # TODO: Hyperplane visualization in the hidden space
    W2 = mlp.W2.flatten()
    b2 = mlp.b2.flatten()[0]
    # Create a grid in hidden space
    h1_range = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 10)
    h2_range = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 10)
    H1, H2 = np.meshgrid(h1_range, h2_range)
    # Solve for H3
    if W2[2] != 0:
        H3 = (-W2[0] * H1 - W2[1] * H2 - b2) / W2[2]
        ax_hidden.plot_surface(H1, H2, H3, alpha=0.3)

    # TODO: Distorted input space transformed by the hidden layer
    # (Optional: Implement if desired)

    # TODO: Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.forward(grid)
    probs = probs.reshape(xx.shape)
    ax_input.contourf(
        xx, yy, probs, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red']
    )
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_input.set_title('Decision Boundary in Input Space')
    ax_input.set_xlabel('X1')
    ax_input.set_ylabel('X2')

    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    ax_gradient.axis('off')
    # Positions of nodes
    input_nodes = [(0, i) for i in range(2)]
    hidden_nodes = [(1, i) for i in range(3)]
    output_node = [(2, 0)]

    # Plot nodes
    for pos in input_nodes:
        ax_gradient.scatter(pos[0], pos[1], s=100, color='blue')
    for pos in hidden_nodes:
        ax_gradient.scatter(pos[0], pos[1], s=100, color='green')
    for pos in output_node:
        ax_gradient.scatter(pos[0], pos[1], s=100, color='red')

    # Plot edges from input to hidden
    dW1_abs = np.abs(mlp.grads['dW1'])
    max_dW1 = np.max(dW1_abs)
    for i in range(2):
        for j in range(3):
            x_values = [input_nodes[i][0], hidden_nodes[j][0]]
            y_values = [input_nodes[i][1], hidden_nodes[j][1]]
            weight_gradient = dW1_abs[i, j] / max_dW1 if max_dW1 != 0 else 0
            linewidth = 1 + weight_gradient * 5
            ax_gradient.plot(
                x_values, y_values, linewidth=linewidth, color='gray'
            )

    # Plot edges from hidden to output
    dW2_abs = np.abs(mlp.grads['dW2'])
    max_dW2 = np.max(dW2_abs)
    for j in range(3):
        x_values = [hidden_nodes[j][0], output_node[0][0]]
        y_values = [hidden_nodes[j][1], output_node[0][1]]
        weight_gradient = dW2_abs[j, 0] / max_dW2 if max_dW2 != 0 else 0
        linewidth = 1 + weight_gradient * 5
        ax_gradient.plot(
            x_values, y_values, linewidth=linewidth, color='gray'
        )

    ax_gradient.set_title('Network Gradients')


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)