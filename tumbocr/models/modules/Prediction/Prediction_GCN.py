from torch.nn import Parameter
import torch
import torch.nn as nn

def gen_A(concur, sums, threshold=0.5, p=0.25, eps=1e-6):
    num_attribute = len(sums)
    sums = np.expand_dims(sums, axis=1)
    _adj = concur / sums
    _adj[_adj < threshold] = 0
    _adj[_adj >= threshold] = 1
    _adj = _adj * p / (_adj.sum(0, keepdims=True) + eps)
    _adj = _adj + (np.identity(num_attribute, np.int) * (1-p))
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class Prediction_GCN(nn.Module):
    def __init__(self, num_hidden, num_classes,in_channels=200, t=0, adj_file=None, inp=None):
        super().__init__()
        self.gc1 = GraphConvolution(in_channels, num_hidden)
        self.gc2 = GraphConvolution(num_hidden, num_hidden)
        # _adj = gen_A(num_classes, t, adj_file)
        # self.A = Parameter(torch.from_numpy(_adj).float())
        self.inp = nn.Parameter(torch.from_numpy(inp).float(), requires_grad=False)

    def forward(self, feature,adj):
        # adj = gen_adj(self.A).detach()
        x = self.gc1(self.inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        x = torch.squeeze(x)
        x = torch.t(x)
        x = torch.matmul(feature, x)

        return x