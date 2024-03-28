import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, activation_func):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = activation_func
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        for layer in self.hidden_layers:
            out = self.activation(layer(out))
        out = self.fc3(out)
        return out


def train(model, criterion, optimizer, X_train, y_train, num_epochs, print_interval=500, plot_loss=False):
    losses = []
    for epoch in range(num_epochs):
        inputs = torch.from_numpy(X_train).float().to(device)
        targets = torch.from_numpy(y_train).float().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch+1) % print_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    if plot_loss:
        # Plotting the loss curve
        plt.plot(np.arange(1, num_epochs+1), losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.show()

    return losses


def test(model, X_test, y_test):
    inputs = torch.from_numpy(X_test).float().to(device)
    targets = torch.from_numpy(y_test).float().to(device)
    outputs = model(inputs)
    return nn.functional.mse_loss(outputs, targets).item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义列名
columns = ['Hidden Size', 'Learning Rate', 'Activation Function',
           'Num Layers', 'Data Size', 'Final Loss', 'Mean Squared Error']

# 创建空的DataFrame
results_df = pd.DataFrame(columns=columns)

input_size = 1
num_classes = 1
num_epochs = 50000
criterion = nn.MSELoss()

# 定义超参数候选列表
# hidden_sizes = [16, 32, 64, 128]
hidden_sizes = [32]
# learning_rates = [0.001, 0.01, 0.1]
learning_rates = [0.001]
activation_functions = [nn.ReLU(), nn.Tanh(), nn.Sigmoid(),
                        nn.ELU(), nn.Softplus()]
# activation_functions = [nn.Softplus()]
# num_layers_list = [3, 5, 7, 9]
num_layers_list = [7]
# data_sizes = [200, 2000, 10000]
data_sizes = [2000]

best_mse = float('inf')
best_model = None
best_hyperparameters = None

with open('output.txt', 'w') as f:
    for hidden_size in hidden_sizes:
        for lr in learning_rates:
            for activation_func in activation_functions:
                for num_layers in num_layers_list:
                    for data_size in data_sizes:
                        np.random.seed(0)
                        X = np.random.uniform(1, 16, (data_size, 1))
                        y = np.log2(X) + np.cos(np.pi * X / 2)
                        train_size = int(0.8 * len(X))
                        valid_size = test_size = int(0.1 * len(X))
                        X_train, X_valid, X_test = X[:train_size], X[train_size:train_size +
                                                                     valid_size], X[train_size + valid_size:]
                        y_train, y_valid, y_test = y[:train_size], y[train_size:train_size +
                                                                     valid_size], y[train_size + valid_size:]

                        model = FeedForwardNN(
                            input_size, hidden_size, num_classes, num_layers, activation_func).to(device)
                        optimizer = optim.Adam(model.parameters(), lr=lr)

                        loss = train(model, criterion, optimizer,
                                     X_train, y_train, num_epochs)[-1]
                        mse = test(model, X_valid, y_valid)

                        print(
                            f"Current Hyperparameters - Hidden Size: {hidden_size}, Learning Rate: {lr}, Activation Function: {activation_func}, Num Layers: {num_layers}, Data Size: {data_size}")
                        print(f"Mean Squared Error on Validation Set: {mse}")
                        results_df = results_df._append({'Hidden Size': hidden_size,
                                                        'Learning Rate': lr,
                                                         'Activation Function': activation_func,
                                                         'Num Layers': num_layers,
                                                         'Data Size': data_size,
                                                         'Final Loss': loss,
                                                         'Mean Squared Error': mse},
                                                        ignore_index=True)
                        output_str = (
                            f"Current Hyperparameters - Hidden Size: {hidden_size}, Learning Rate: {lr}, Activation Function: {activation_func}, Num Layers: {num_layers}, Data Size: {data_size}\n"
                            f"Mean Squared Error on Validation Set: {mse}\n\n"
                        )
                        f.write(output_str)

                        if mse < best_mse:
                            best_mse = mse
                            best_model = model
                            best_hyperparameters = (
                                hidden_size, lr, activation_func, num_layers, data_size)

                        plt.scatter(X_valid, y_valid, color='blue',
                                    label='Original Data')

                        plt.scatter(X_valid, model(torch.from_numpy(X_valid).float().to(
                            device)).detach().cpu().numpy(), color='red', label='Predicted Data')
                        plt.xlabel('X')
                        plt.ylabel('y')
                        plt.title('Validation Set - Original vs Predicted')
                        plt.legend()
                        filename = f"{hidden_size}_{lr}_{activation_func}_{num_layers}_{data_size}.png"
                        plt.savefig(filename)
                        plt.close()

    print(
        f"Best Hyperparameters - Hidden Size: {best_hyperparameters[0]}, Learning Rate: {best_hyperparameters[1]}, Activation Function: {best_hyperparameters[2]}, Num Layers: {best_hyperparameters[3]}, Data Size: {best_hyperparameters[4]}")
    print(f"Best Mean Squared Error on Validation Set: {best_mse}")
    f.write(
        f"Best Hyperparameters - Hidden Size: {best_hyperparameters[0]}, Learning Rate: {best_hyperparameters[1]}, Activation Function: {best_hyperparameters[2]}, Num Layers: {best_hyperparameters[3]}, Data Size: {best_hyperparameters[4]}\n")
    f.write(f"Best Mean Squared Error on Validation Set: {best_mse}\n")

# 保存训练超参数结果
results_df.to_csv('hyperparameter_results.csv', index=False)

# 使用最佳超参数重新训练模型
hidden_size, lr, activation_func, num_layers, data_size = best_hyperparameters
np.random.seed(0)
X = np.random.uniform(1, 16, (data_size, 1))
y = np.log2(X) + np.cos(np.pi * X / 2)
train_size = int(0.8 * len(X))
valid_size = test_size = int(0.1 * len(X))
X_train, X_valid, X_test = X[:train_size], X[train_size:train_size +
                                             valid_size], X[train_size + valid_size:]
y_train, y_valid, y_test = y[:train_size], y[train_size:train_size +
                                             valid_size], y[train_size + valid_size:]
final_model = FeedForwardNN(input_size, hidden_size,
                            num_classes, num_layers, activation_func).to(device)
optimizer = optim.Adam(final_model.parameters(), lr=best_hyperparameters[1])
train(final_model, criterion, optimizer, X_train,
      y_train, num_epochs, plot_loss=True)

# 在测试集上测试最终模型
test_mse = test(final_model, X_test, y_test)
print(f"Mean Squared Error on Test Set: {test_mse}")

# 可视化
plt.scatter(X_test, y_test, color='blue', label='Original Data')
plt.scatter(X_test, final_model(torch.from_numpy(X_test).float().to(device)
                                ).detach().cpu().numpy(), color='red', label='Predictions')
plt.legend()
plt.show()
