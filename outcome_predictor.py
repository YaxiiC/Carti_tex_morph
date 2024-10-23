import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return torch.sigmoid(outputs)

def train_logistic_regression_model(features, labels, num_epochs=10000, learning_rate=0.001):
    input_dim = features.size(1)
    output_dim = labels.size(1)

    model = LogisticRegressionModel(input_dim, output_dim)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for logistic regression
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def output_feature_importance(model):
    # Access the learned weights as the feature importance
    importance = model.linear.weight.detach().cpu().numpy()  # Convert the weights to numpy for inspection
    print("Feature Importance:", importance)
    return importance

