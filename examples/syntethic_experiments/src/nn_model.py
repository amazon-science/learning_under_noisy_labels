import torch
from torch import nn

class FeedForwardNN(nn.Module):
    """
    A feed-forward neural network with one hidden layer.
    """
    def __init__(self, num_features: int, hidden_dim: int, num_labels: int) -> None:
        """
        Initialize the feed-forward neural network.

        Args:
            num_features (int): The number of input features.
            hidden_dim (int): The number of units in the hidden layer.
            num_labels (int): The number of output labels (classes).
        """
        super().__init__()
        assert num_labels > 2
        self.input_layer = nn.Linear(num_features, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, num_labels)

    def forward(self, inp_features):
        """
        Forward pass of the neural network.

        Args:
            inp_features (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The output logits.
        """
        out = self.input_layer(inp_features)
        out = nn.functional.tanh(out)
        out = self.out_layer(out)
        return out

class LogisticRegression(nn.Module):
    """
    A logistic regression model for binary classification.
    """
    def __init__(self, num_features: int, hidden_dim: int, num_labels: int) -> None:
        """
        Initialize the logistic regression model.

        Args:
            num_features (int): The number of input features.
            hidden_dim (int): Not used, but kept for consistency with FeedForwardNN.
            num_labels (int): The number of output labels (classes).
        """
        super().__init__()
        assert num_labels == 2
        self.linear = nn.Linear(num_features, num_labels)

    def forward(self, inp_features):
        """
        Forward pass of the logistic regression model.

        Args:
            inp_features (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The output logits.
        """
        out = self.linear(inp_features)
        out = nn.functional.tanh(out)
        return out

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    num_features, num_classes, hidden_dim, batch_size = 2, 4, 4, 32
    input_vector = torch.randn((batch_size, num_features), device=device)
    model = FeedForwardNN(num_features, hidden_dim, num_classes).to(device).eval()
    with torch.no_grad():
        output = model(input_vector)
        print(output.shape)