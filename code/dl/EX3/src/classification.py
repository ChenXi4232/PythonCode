from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
from torch import optim
import matplotlib.pyplot as plt
from dgl.transforms import DropEdge
from dgl.transforms import add_self_loop
import dgl.function as fn
import torch.nn.functional as F
import torch.nn as nn
import torch
import dgl
from dgl.data import citation_graph as citegrh
import numpy as np


seed = 24
dgl.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cora_dataset = citegrh.load_cora(raw_dir="./dataset/cora/")
cora_graph = cora_dataset[0]
cora_graph = cora_graph.to(device)
cora_features = cora_graph.ndata['feat']
cora_labels = cora_graph.ndata['label']
cora_train_mask = cora_graph.ndata['train_mask']
cora_val_mask = cora_graph.ndata['val_mask']
cora_test_mask = cora_graph.ndata['test_mask']

citeseer_dataset = citegrh.load_citeseer(raw_dir="./dataset/citeseer/")
citeseer_graph = citeseer_dataset[0]
citeseer_graph = citeseer_graph.to(device)
citeseer_features = citeseer_graph.ndata['feat']
citeseer_labels = citeseer_graph.ndata['label']
citeseer_train_mask = citeseer_graph.ndata['train_mask']
citeseer_val_mask = citeseer_graph.ndata['val_mask']
citeseer_test_mask = citeseer_graph.ndata['test_mask']


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)  # 初始化权重
        nn.init.zeros_(self.bias)  # 初始化偏置

    def forward(self, g, features, drop_edge_rate=0.0, with_loop=True):

        if drop_edge_rate > 0:
            g = DropEdge(drop_edge_rate)(g)

        with g.local_scope():
            if with_loop:
                g = add_self_loop(g)
            g.ndata['h'] = features
            g.update_all(message_func=fn.copy_u('h', 'm'),
                         reduce_func=fn.sum('m', 'h'))
            h = g.ndata['h']
            h = torch.matmul(h, self.weight) + self.bias
            return h


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers, activation=F.relu, dropout=0.8, drop_edge_rate=0.0, with_loop=True, pair_mode='PN-SCS'):
        super(GCN, self).__init__() b
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.norm = PairNorm(mode=pair_mode, scale=1)
        self.drop_edge_rate = drop_edge_rate
        self.with_loop = with_loop

        # 输入层
        self.layers.append(GraphConvolution(in_features, hidden_features))
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(
                hidden_features, hidden_features))
        # 输出层
        self.layers.append(GraphConvolution(hidden_features, out_features))

    def forward(self, g, features):
        h = features
        for layer in self.layers[:-1]:
            h = self.norm(h)
            h = self.activation(
                layer(g, h, self.drop_edge_rate, with_loop=self.with_loop))
            h = self.dropout(h)
        h = self.layers[-1](g, h)
        output = F.log_softmax(h, dim=1)
        return output

    def evaluate(self, g, features):
        with torch.no_grad():
            h = features
            for layer in self.layers[:-1]:
                h = self.norm(h)
                h = self.activation(layer(g, h))
            h = self.layers[-1](g, h)
            output = F.log_softmax(h, dim=1)
        return output


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization 
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version

            ('SCS'-mode is not in the paper but we found it works well in practice, 
              especially for GCN and GAT.)

            PairNorm is typically used after each graph convolution operation. 
        """
        assert mode in ['None', 'PN',  'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (
                1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (
                1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


def plot_metrics(train_losses, train_metrics, val_losses, val_metrics, metric_name, save_path="metrics.png"):
    """
    根据训练集损失、训练集评价指标、验证集损失和验证集评价指标绘制图像。
    使用左右两个纵坐标轴分别显示损失和评价指标。

    参数:
    - train_losses: list - 训练集损失
    - train_metrics: list - 训练集评价指标
    - val_losses: list - 验证集损失
    - val_metrics: list - 验证集评价指标
    - metric_name: str - 评价指标的名称
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 绘制训练集和验证集的损失曲线
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')

    # 使用 twinx 方法创建共享 x 轴的第二个 y 轴
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_metrics, 'g-', label=f'Training {metric_name}')
    ax2.plot(epochs, val_metrics, 'm-', label=f'Validation {metric_name}')
    ax2.set_ylabel(metric_name, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title(f'Training and Validation Loss and {metric_name}')
    plt.savefig(save_path)
    plt.show()


def train_model(model, g, features, labels, train_mask, val_mask, epochs=100):

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    sheduler = StepLR(optimizer, step_size=0.3*epochs, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    total_train_loss = []
    total_train_acc = []
    total_val_loss = []
    total_val_acc = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(g, features)
        loss = criterion(output[train_mask], labels[train_mask])
        loss.backward()
        acc = accuracy_score(labels[train_mask].cpu(), torch.max(
            output[train_mask], dim=1)[1].cpu())
        total_train_loss.append(loss.item())
        total_train_acc.append(acc)
        optimizer.step()
        sheduler.step()

        # 在验证集上计算损失
        model.eval()
        with torch.no_grad():
            val_output = model.evaluate(g, features)
            val_loss = criterion(val_output[val_mask], labels[val_mask])
            val_acc = accuracy_score(labels[val_mask].cpu(), torch.max(
                val_output[val_mask], dim=1)[1].cpu())
            total_val_loss.append(val_loss.item())
            total_val_acc.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), './best_model.pth')

        print(
            f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc:.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}')

    print('最佳验证集准确率: ', best_val_acc)
    print('训练完成。')

    return total_train_loss, total_train_acc, total_val_loss, total_val_acc


# 调参分析，只在 cora 数据集上进行
# 自环对层数的影响，分析层数为 2，8，16, 32 时的损失和准确率，不使用 PairNorm 和 DropEdge
layers = [2, 8, 16, 32]
with_loop = [True, False]

for layer in layers:
    for loop in with_loop:
        model = GCN(in_features=cora_features.shape[1],
                    hidden_features=256,
                    out_features=cora_labels.max().item() + 1,
                    num_layers=layer,
                    activation=F.relu,
                    drop_edge_rate=0,
                    with_loop=loop,
                    pair_mode="None").to(device)

        total_train_loss, total_train_acc, total_val_loss, total_val_acc = train_model(
            model, cora_graph, cora_features, cora_labels, cora_train_mask, cora_val_mask, epochs=400)
        plot_metrics(total_train_loss, total_train_acc, total_val_loss, total_val_acc,
                     'Accuracy', save_path=f'classify_layer_{layer}_loop_{loop}.png')

# PairNorm 对层数的影响，分析层数为 2，8，16, 32 时的损失和准确率，不使用 DropEdge、自环
layers = [2, 8, 16, 32]
pair_modes = ['PN', 'PN-SI', 'PN-SCS']

for layer in layers:
    for pair_mode in pair_modes:
        model = GCN(in_features=cora_features.shape[1],
                    hidden_features=256,
                    out_features=cora_labels.max().item() + 1,
                    num_layers=layer,
                    activation=F.relu,
                    drop_edge_rate=0,
                    with_loop=False,
                    pair_mode=pair_mode).to(device)

        total_train_loss, total_train_acc, total_val_loss, total_val_acc = train_model(
            model, cora_graph, cora_features, cora_labels, cora_train_mask, cora_val_mask, epochs=400)
        plot_metrics(total_train_loss, total_train_acc, total_val_loss, total_val_acc,
                     'Accuracy', save_path=f'classify_layer_{layer}_pair_{pair_mode}.png')

# DropEdge 对层数的影响，分析层数为 2，8，16，32 时的损失和准确率，不使用 PairNorm、自环
layers = [2, 8, 16, 32]
drop_edge_rates = [0.5, 0.75, 0.9]

for layer in layers:
    for drop_edge_rate in drop_edge_rates:
        model = GCN(in_features=cora_features.shape[1],
                    hidden_features=256,
                    out_features=cora_labels.max().item() + 1,
                    num_layers=layer,
                    activation=F.relu,
                    drop_edge_rate=drop_edge_rate,
                    with_loop=False,
                    pair_mode="None").to(device)

        total_train_loss, total_train_acc, total_val_loss, total_val_acc = train_model(
            model, cora_graph, cora_features, cora_labels, cora_train_mask, cora_val_mask, epochs=400)
        plot_metrics(total_train_loss, total_train_acc, total_val_loss, total_val_acc,
                     'Accuracy', save_path=f'classify_layer_{layer}_drop_{drop_edge_rate}.png')

# 合适的 DropEdge 、PairNorm 和自环参数下，在不同层数下的损失和准确率
layers = [2, 8, 16, 32]

for layer in layers:
    model = GCN(in_features=cora_features.shape[1],
                hidden_features=256,
                out_features=cora_labels.max().item() + 1,
                num_layers=layer,
                activation=F.relu,
                drop_edge_rate=0.85,
                with_loop=True,
                pair_mode="PN-SCS").to(device)

    total_train_loss, total_train_acc, total_val_loss, total_val_acc = train_model(
        model, cora_graph, cora_features, cora_labels, cora_train_mask, cora_val_mask, epochs=400)
    plot_metrics(total_train_loss, total_train_acc, total_val_loss,
                 total_val_acc, 'Accuracy', save_path=f'classify_layer_{layer}_best.png')

# 合适的 DropEdge 、PairNorm 、层数和自环参数下，不同激活函数的效果
activations = [F.relu, F.sigmoid, F.tanh, lambda x: x]

for activation in activations:
    model = GCN(in_features=cora_features.shape[1],
                hidden_features=256,
                out_features=cora_labels.max().item() + 1,
                num_layers=2,
                activation=activation,
                drop_edge_rate=0.85,
                with_loop=True,
                pair_mode="PN-SCS").to(device)

    total_train_loss, total_train_acc, total_val_loss, total_val_acc = train_model(
        model, cora_graph, cora_features, cora_labels, cora_train_mask, cora_val_mask, epochs=100)
    if activation.__name__ == '<lambda>':
        plot_metrics(total_train_loss, total_train_acc, total_val_loss, total_val_acc,
                     'Accuracy', save_path=f'classify_layer2_with_loop_PairNorm_DropEdge_linear.png')
    else:
        plot_metrics(total_train_loss, total_train_acc, total_val_loss, total_val_acc, 'Accuracy',
                     save_path=f'classify_layer2_with_loop_PairNorm_DropEdge_{activation.__name__}.png')

model = GCN(in_features=cora_features.shape[1],
            hidden_features=256,
            out_features=cora_labels.max().item() + 1,
            num_layers=2,
            activation=F.relu,
            drop_edge_rate=0.85,
            with_loop=True,
            pair_mode="PN-SCS").to(device)


total_train_loss, total_train_acc, total_val_loss, total_val_acc = train_model(
    model, cora_graph, cora_features, cora_labels, cora_train_mask, cora_val_mask, epochs=200)
plot_metrics(total_train_loss, total_train_acc, total_val_loss,
             total_val_acc, 'Accuracy', save_path='cora_classify_metrics.png')

final_model = GCN(in_features=cora_features.shape[1],
                  hidden_features=256,
                  out_features=cora_labels.max().item() + 1,
                  num_layers=2,
                  activation=F.relu,
                  drop_edge_rate=0.85,
                  with_loop=True,
                  pair_mode="PN-SCS").to(device)

train_model(final_model, cora_graph, cora_features, cora_labels,
            cora_train_mask, cora_test_mask, epochs=200)

final_model.eval()
with torch.no_grad():
    test_output = final_model.evaluate(cora_graph, cora_features)
    test_loss = nn.CrossEntropyLoss()(
        test_output[cora_test_mask], cora_labels[cora_test_mask])
    test_acc = accuracy_score(cora_labels[cora_test_mask].cpu(
    ), torch.max(test_output[cora_test_mask], dim=1)[1].cpu())
    print(f"测试集损失: {test_loss.item():.4f}, 测试集准确率: {test_acc:.4f}")

model = GCN(in_features=citeseer_features.shape[1],
            hidden_features=256,
            out_features=citeseer_labels.max().item() + 1,
            num_layers=2,
            activation=F.relu,
            drop_edge_rate=1,
            with_loop=True,
            pair_mode="PN-SCS").to(device)

total_train_loss, total_train_acc, total_val_loss, total_val_acc = train_model(
    model, citeseer_graph, citeseer_features, citeseer_labels, citeseer_train_mask, citeseer_val_mask, epochs=200)
plot_metrics(total_train_loss, total_train_acc, total_val_loss,
             total_val_acc, 'Accuracy', save_path='citeseer_classify_metrics.png')

final_model = GCN(in_features=citeseer_features.shape[1],
                  hidden_features=256,
                  out_features=citeseer_labels.max().item() + 1,
                  num_layers=2,
                  activation=F.relu,
                  drop_edge_rate=1,
                  with_loop=True,
                  pair_mode="PN-SCS").to(device)

train_model(final_model, citeseer_graph, citeseer_features,
            citeseer_labels, citeseer_train_mask, citeseer_test_mask, epochs=200)

final_model.eval()
with torch.no_grad():
    test_output = final_model.evaluate(citeseer_graph, citeseer_features)
    test_loss = nn.CrossEntropyLoss()(
        test_output[citeseer_test_mask], citeseer_labels[citeseer_test_mask])
    test_acc = accuracy_score(citeseer_labels[citeseer_test_mask].cpu(
    ), torch.max(test_output[citeseer_test_mask], dim=1)[1].cpu())
    print(f"测试集损失: {test_loss.item():.4f}, 测试集准确率: {test_acc:.4f}")
