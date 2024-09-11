import torch.nn as nn

class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        x = self.linear(x)
        return x
    
