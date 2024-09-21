import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.reset_parameters()

    def forward(self, x):
        return self.linear(x)

    def reset_parameters(self):
        self.linear.reset_parameters()