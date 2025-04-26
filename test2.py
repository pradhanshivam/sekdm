import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data

class Args:
    def __init__(self):
        self.input_dim = 8
        self.hidden_dim = 64
        self.output_dim = 2
        self.lr = 0.001
        self.weight_decay = 5e-4
        self.epochs = 100
        self.device = torch.device('cpu')  # <-- Yahi fix kiya hai

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def train(model, optimizer, criterion, data, device):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, device):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = (pred == data.y).sum().item()
        acc = correct / data.num_nodes
    return acc

def main():
    args = Args()

    # Example dummy data
    num_nodes = 100
    x = torch.randn((num_nodes, args.input_dim))
    edge_index = torch.randint(0, num_nodes, (2, 500))
    y = torch.randint(0, args.output_dim, (num_nodes,))

    data = pyg_data.Data(x=x, edge_index=edge_index, y=y).to(args.device)

    model = GNNModel(args.input_dim, args.hidden_dim, args.output_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        loss = train(model, optimizer, criterion, data, args.device)
        acc = test(model, data, args.device)
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

if __name__ == "__main__":
    main()
