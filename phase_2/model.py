import torch.nn as nn

class Iris2LayerClassifier(nn.Module):
    def __init__(self):
        super(Iris2LayerClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
    