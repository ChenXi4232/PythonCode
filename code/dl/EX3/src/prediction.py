from torch.optim.lr_scheduler import StepLR
import itertools
from torch import optim
from sklearn.metrics import roc_auc_score
import dgl.function as fn
import torch.nn.functional as F
from dgl.transforms import DropEdge
from dgl.transforms import add_self_loop
import torch.nn as nn
import dgl
from dgl.data import citation_graph as citegrh
import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


seed = 24
dgl.random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def load_cora_dataset(train_ratio, val_ratio):
    cora_dataset = citegrh.load_cora(raw_dir="./dataset/cora/")

    cora_graph = cora_dataset[0].to(device)
    cora_features = cora_graph.ndata['feat']

    u, v = cora_graph.edges()
    eids = np.arange(cora_graph.number_of_edges())
    eids = np.random.permutation(eids)  # 将顺序打乱

    train_size = int(cora_graph.number_of_edges() * train_ratio)
    val_size = int(cora_graph.number_of_edges() * val_ratio)
    test_size = cora_graph.number_of_edges() - train_size - val_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size+val_size]
                             ], v[eids[test_size:test_size+val_size]]
    train_pos_u, train_pos_v = u[eids[test_size+val_size:]
                                 ], v[eids[test_size+val_size:]]

    adj = sp.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())))
    adj_neg = 1 - adj.todense() - np.eye(cora_graph.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), cora_graph.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]
                                   ], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size+val_size]
                                 ], neg_v[neg_eids[test_size:test_size+val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size+val_size:]
                                     ], neg_v[neg_eids[test_size+val_size:]]

    train_pos_g = dgl.graph((train_pos_u, train_pos_v),
                            num_nodes=cora_graph.number_of_nodes()).to(device)
    train_neg_g = dgl.graph((train_neg_u, train_neg_v),
                            num_nodes=cora_graph.number_of_nodes()).to(device)
    val_pos_g = dgl.graph((val_pos_u, val_pos_v),
                          num_nodes=cora_graph.number_of_nodes()).to(device)
    val_neg_g = dgl.graph((val_neg_u, val_neg_v),
                          num_nodes=cora_graph.number_of_nodes()).to(device)
    test_pos_g = dgl.graph((test_pos_u, test_pos_v),
                           num_nodes=cora_graph.number_of_nodes()).to(device)
    test_neg_g = dgl.graph((test_neg_u, test_neg_v),
                           num_nodes=cora_graph.number_of_nodes()).to(device)

    return cora_graph, cora_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g


def load_citeseer_dataset(train_ratio, val_ratio):
    citeseer_dataset = citegrh.load_citeseer(raw_dir="./dataset/citeseer/")

    citeseer_graph = citeseer_dataset[0].to(device)
    citeseer_features = citeseer_graph.ndata['feat']

    u, v = citeseer_graph.edges()
    eids = np.arange(citeseer_graph.number_of_edges())
    eids = np.random.permutation(eids)  # 将顺序打乱

    train_size = int(citeseer_graph.number_of_edges() * train_ratio)
    val_size = int(citeseer_graph.number_of_edges() * val_ratio)
    test_size = citeseer_graph.number_of_edges() - train_size - val_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    val_pos_u, val_pos_v = u[eids[test_size:test_size+val_size]
                             ], v[eids[test_size:test_size+val_size]]
    train_pos_u, train_pos_v = u[eids[test_size+val_size:]
                                 ], v[eids[test_size+val_size:]]

    adj = sp.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())))
    adj_neg = 1 - adj.todense() - np.eye(citeseer_graph.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), citeseer_graph.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]
                                   ], neg_v[neg_eids[:test_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[test_size:test_size+val_size]
                                 ], neg_v[neg_eids[test_size:test_size+val_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size+val_size:]
                                     ], neg_v[neg_eids[test_size+val_size:]]

    train_pos_g = dgl.graph((train_pos_u, train_pos_v),
                            num_nodes=citeseer_graph.number_of_nodes()).to(device)
    train_neg_g = dgl.graph((train_neg_u, train_neg_v),
                            num_nodes=citeseer_graph.number_of_nodes()).to(device)
    val_pos_g = dgl.graph((val_pos_u, val_pos_v),
                          num_nodes=citeseer_graph.number_of_nodes()).to(device)
    val_neg_g = dgl.graph((val_neg_u, val_neg_v),
                          num_nodes=citeseer_graph.number_of_nodes()).to(device)
    test_pos_g = dgl.graph((test_pos_u, test_pos_v),
                           num_nodes=citeseer_graph.number_of_nodes()).to(device)
    test_neg_g = dgl.graph((test_neg_u, test_neg_v),
                           num_nodes=citeseer_graph.number_of_nodes()).to(device)

    return citeseer_graph, citeseer_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g


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
        super(GCN, self).__init__()
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
        return h

    def evaluate(self, g, features):
        with torch.no_grad():
            h = features
            for layer in self.layers[:-1]:
                h = self.norm(h)
                h = self.activation(layer(g, h))
            h = self.layers[-1](g, h)
        return h


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


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))

            return g.edata['score'][:, 0]


def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat([torch.ones(pos_score.shape[0]),
                       torch.zeros(neg_score.shape[0])]).to(device)
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).cpu().detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).cpu().detach().numpy()
    return roc_auc_score(labels, scores)


def train_model(model, g, features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, pred, epochs=500):

    optimizer = optim.Adam(itertools.chain(
        model.parameters(), pred.parameters()), lr=0.01, weight_decay=5e-3)
    scheduler = StepLR(optimizer, step_size=0.3*epochs, gamma=0.5)

    best_auc = 0
    total_train_loss = []
    total_train_auc = []
    total_val_loss = []
    total_val_auc = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        h = model(g, features)
        pos_score = pred(train_pos_g, h)
        neg_score = pred(train_neg_g, h)
        loss = compute_loss(pos_score, neg_score)
        acc = compute_auc(pos_score, neg_score)
        total_train_loss.append(loss.item())
        total_train_auc.append(acc)

        loss.backward()
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            val_pos_score = pred(val_pos_g, h)
            val_neg_score = pred(val_neg_g, h)
            val_loss = compute_loss(val_pos_score, val_neg_score)
            val_auc = compute_auc(val_pos_score, val_neg_score)
            total_val_loss.append(val_loss.item())
            total_val_auc.append(val_auc)
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), './best_model.pth')
        if epoch % 5 == 0:
            print(
                f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc:.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_auc:.4f}')

    print('最佳验证集AUC: ', best_auc)
    print('训练完成。')

    return total_train_loss, total_train_auc, total_val_loss, total_val_auc


# 超参数分析

cora_graph, cora_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = load_cora_dataset(
    0.7, 0.15)

# 分析不使用 PairNorm 和 DropEdge 时，自环对不同层数编码器的影响
layers = [2, 8, 16, 32]
with_loop = [True, False]

for layer in layers:
    for loop in with_loop:
        model = GCN(in_features=cora_features.shape[1],
                    hidden_features=256,
                    out_features=256,
                    num_layers=layer,
                    activation=F.relu,
                    drop_edge_rate=0,
                    with_loop=loop,
                    pair_mode="None").to(device)

        pred = DotPredictor().to(device)

        total_train_loss, total_train_auc, total_val_loss, total_val_auc = train_model(
            model, cora_graph, cora_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, pred, epochs=500)
        plot_metrics(total_train_loss, total_train_auc, total_val_loss,
                     total_val_auc, 'AUC', save_path=f'predict_layer_{layer}_loop_{loop}.png')

# 分析不使用 Dropedge 和自环时，Pair_Norm 对不同层数编码器的影响
layers = [2, 8, 16, 32]
pair_modes = ['PN', 'PN-SI', 'PN-SCS']

for layer in layers:
    for pair_mode in pair_modes:
        model = GCN(in_features=cora_features.shape[1],
                    hidden_features=256,
                    out_features=256,
                    num_layers=layer,
                    activation=F.relu,
                    drop_edge_rate=0,
                    with_loop=False,
                    pair_mode=pair_mode).to(device)

        pred = DotPredictor().to(device)

        total_train_loss, total_train_auc, total_val_loss, total_val_auc = train_model(
            model, cora_graph, cora_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, pred, epochs=500)
        plot_metrics(total_train_loss, total_train_auc, total_val_loss, total_val_auc,
                     'AUC', save_path=f'predict_layer_{layer}_pair_{pair_mode}.png')

# 分析不使用 PairNorm 和自环时，DropEdge 对不同层数编码器的影响
layers = [2, 8, 16, 32]
drop_edge_rates = [0.5, 0.75, 0.9]

for layer in layers:
    for drop_edge_rate in drop_edge_rates:
        model = GCN(in_features=cora_features.shape[1],
                    hidden_features=256,
                    out_features=256,
                    num_layers=layer,
                    activation=F.relu,
                    drop_edge_rate=drop_edge_rate,
                    with_loop=False,
                    pair_mode="None").to(device)

        pred = DotPredictor().to(device)

        total_train_loss, total_train_auc, total_val_loss, total_val_auc = train_model(
            model, cora_graph, cora_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, pred, epochs=500)
        plot_metrics(total_train_loss, total_train_auc, total_val_loss, total_val_auc,
                     'AUC', save_path=f'predict_layer_{layer}_drop_{drop_edge_rate}.png')

# 分析合适的 DropEdge、PairNorm 和自环参数下，不同层数编码器的效果
layers = [2, 8, 16, 32]

for layer in layers:
    model = GCN(in_features=cora_features.shape[1],
                hidden_features=256,
                out_features=256,
                num_layers=layer,
                activation=F.relu,
                drop_edge_rate=0.85,
                with_loop=True,
                pair_mode="PN-SCS").to(device)

    pred = DotPredictor().to(device)

    total_train_loss, total_train_auc, total_val_loss, total_val_auc = train_model(
        model, cora_graph, cora_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, pred, epochs=500)
    plot_metrics(total_train_loss, total_train_auc, total_val_loss,
                 total_val_auc, 'AUC', save_path=f'predict_layer_{layer}_best.png')

# 分析合适的 DropEdge、PairNorm、层数和自环参数下，不同激活函数的效果
activations = [F.relu, F.sigmoid, F.tanh, lambda x: x]

for activation in activations:
    model = GCN(in_features=cora_features.shape[1],
                hidden_features=256,
                out_features=256,
                num_layers=2,
                activation=activation,
                drop_edge_rate=0.85,
                with_loop=True,
                pair_mode="PN-SCS").to(device)

    pred = DotPredictor().to(device)

    total_train_loss, total_train_auc, total_val_loss, total_val_auc = train_model(
        model, cora_graph, cora_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, pred, epochs=500)
    if activation.__name__ == '<lambda>':
        plot_metrics(total_train_loss, total_train_auc, total_val_loss, total_val_auc,
                     'AUC', save_path=f'predict_layer2_with_loop_PairNorm_DropEdge_linear.png')
    else:
        plot_metrics(total_train_loss, total_train_auc, total_val_loss, total_val_auc, 'AUC',
                     save_path=f'predict_layer2_with_loop_PairNorm_DropEdge_{activation.__name__}.png')

cora_graph, cora_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = load_cora_dataset(
    0.7, 0.15)
model = GCN(in_features=cora_features.shape[1],
            hidden_features=256,
            out_features=256,
            num_layers=2,
            activation=F.relu,
            drop_edge_rate=0.85,
            with_loop=True,
            pair_mode="PN-SCS").to(device)

pred = DotPredictor().to(device)

total_train_loss, total_train_auc, total_val_loss, total_val_auc = train_model(
    model, cora_graph, cora_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, pred, epochs=500)
plot_metrics(total_train_loss, total_train_auc, total_val_loss,
             total_val_auc, 'AUC', save_path='citeseer_predict_metrics.png')

# 在测试集上测试
final_model = GCN(in_features=cora_features.shape[1],
                  hidden_features=256,
                  out_features=256,
                  num_layers=2,
                  activation=F.relu,
                  drop_edge_rate=0.85,
                  with_loop=True,
                  pair_mode="PN-SCS").to(device)

total_train_loss, total_train_auc, total_val_loss, total_val_auc = train_model(
    final_model, cora_graph, cora_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, pred, epochs=500)

with torch.no_grad():
    h = final_model(cora_graph, cora_features)
    test_pos_score = pred(test_pos_g, h)
    test_neg_score = pred(test_neg_g, h)
    test_auc = compute_auc(test_pos_score, test_neg_score)
    test_loss = compute_loss(test_pos_score, test_neg_score)
    print(f"测试集AUC: {test_auc:.4f}, 测试集Loss: {test_loss:.4f}")

citeseer_graph, citeseer_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = load_citeseer_dataset(
    0.7, 0.15)

model = GCN(in_features=citeseer_features.shape[1],
            hidden_features=256,
            out_features=256,
            num_layers=2,
            activation=F.relu,
            drop_edge_rate=1,
            with_loop=True,
            pair_mode="PN-SCS").to(device)

pred = DotPredictor().to(device)

total_train_loss, total_train_auc, total_val_loss, total_val_auc = train_model(
    model, citeseer_graph, citeseer_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, pred, epochs=500)
plot_metrics(total_train_loss, total_train_auc, total_val_loss,
             total_val_auc, 'AUC', save_path='citeseer_predict_metrics.png')

# 在测试集上测试
final_model = GCN(in_features=citeseer_features.shape[1],
                  hidden_features=256,
                  out_features=256,
                  num_layers=2,
                  activation=F.relu,
                  drop_edge_rate=1,
                  with_loop=True,
                  pair_mode="PN-SCS").to(device)

total_train_loss, total_train_auc, total_val_loss, total_val_auc = train_model(
    final_model, citeseer_graph, citeseer_features, train_pos_g, train_neg_g, val_pos_g, val_neg_g, pred, epochs=500)

with torch.no_grad():
    h = final_model(citeseer_graph, citeseer_features)
    test_pos_score = pred(test_pos_g, h)
    test_neg_score = pred(test_neg_g, h)
    test_auc = compute_auc(test_pos_score, test_neg_score)
    test_loss = compute_loss(test_pos_score, test_neg_score)
    print(f"测试集AUC: {test_auc:.4f}, 测试集Loss: {test_loss:.4f}")
