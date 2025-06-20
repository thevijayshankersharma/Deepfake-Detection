import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

# 1. Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 2. Create model and print summary
model = MyModel()
summary(model, input_size=(1, 100))  # (batch_size, input_features)

# 3. Dummy training step (optional, to make the model "trained")
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Fake data
inputs = torch.randn(5, 100)       # batch of 5, 100 features
targets = torch.randint(0, 10, (5,))  # batch of 5 labels

# One training step
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

# 4. Save the model weights
torch.save(model.state_dict(), "./model/df_model.pt")

# 5. To load later
loaded_model = MyModel()
loaded_model.load_state_dict(torch.load("./model/df_model.pt"))
loaded_model.eval()

# 6. Confirm it's loaded
print("\n✅ Model Loaded Successfully:")
print(loaded_model)
