import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse
import numpy as np

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A):
        # A: adjacency matrix (batch_size, num_nodes, num_nodes)
        # X: input features (batch_size, num_nodes, in_features)
        out = torch.bmm(A, X)  # Graph convolution: A @ X
        out = self.linear(out)  # Linear transformation
        return F.relu(out)

def batch_size_too_large(batch_size_nodes):
    # 确保这个函数在类的作用域之外，能够被调用
    large_batch_size_threshold = 10000  # 设定适当的阈值
    return batch_size_nodes > large_batch_size_threshold

class GCNLSTM(nn.Module):
    def __init__(self, input_dim, gcn_hidden_dim, lstm_hidden_dim, num_layers, output_dim, adj_matrix,num_nodes):
        super(GCNLSTM, self).__init__()
        # GCN部分
        self.gcn1 = GraphConvolution(input_dim, gcn_hidden_dim)
        self.gcn2 = GraphConvolution(gcn_hidden_dim, gcn_hidden_dim)
        if isinstance(adj_matrix, np.ndarray):
            self.adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            # print(self.adj_matrix)
        else:
            self.adj_matrix = adj_matrix
        self._gcn_hidden_dim = gcn_hidden_dim
        # LSTM部分
        self.lstm = nn.LSTM(input_size=gcn_hidden_dim, hidden_size=lstm_hidden_dim,
                                num_layers=num_layers, batch_first=True)
        # 输出层
        self.fc_out = nn.Linear(lstm_hidden_dim, output_dim)


    def forward(self, X):
        X = X.unsqueeze(-1)
        batch_size, time_steps, num_nodes, input_dim = X.size()

        # 获取输入 X 所在的设备（CPU 或 CUDA）
        device = X.device # cuda:0
        # 创建邻接矩阵并扩展维度
        A = self.adj_matrix.to(device).unsqueeze(0).expand(batch_size, -1, -1)

        # GCN部分（对每个时间步进行处理）
        gcn_out = []
        for t in range(time_steps):
            gcn_feature = self.gcn1(X[:, t, :, :], A) # (batch_size, num_nodes, out_features)
            gcn_feature = self.gcn2(gcn_feature, A)
            gcn_out.append(gcn_feature.unsqueeze(1))
        # 拼接时间维度特征
        gcn_out = torch.cat(gcn_out, dim=1) # (batch_size, time_steps, num_nodes, gcn_hidden_dim)
        # 将 num_nodes 和时间步分开进行 LSTM 处理
        lstm_out = []
        for node in range(num_nodes):
            # 对每个节点的时间序列进行 LSTM 处理
            node_features = gcn_out[:, :, node, :]  # (batch_size, time_steps, gcn_hidden_dim)
            lstm_node_out, _ = self.lstm(node_features)  # (batch_size, time_steps, lstm_hidden_dim)
            lstm_out.append(lstm_node_out[:, -1, :].unsqueeze(1))  # 取最后一个时间步的输出, (batch_size, 1, lstm_hidden_dim)
        # 合并所有节点的 LSTM 输出
        lstm_out = torch.cat(lstm_out, dim=1)  # (batch_size, num_nodes, lstm_hidden_dim)

        # 对每个节点的特征进行预测.
        output = self.fc_out(lstm_out)  # (batch_size, num_nodes, output_dim) [64, 151, 1]
        # print(output.shape)
        # print(output)
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=32)
        parser.add_argument("--num_layers", type=int, default=2)
        return parser

