import torch
import torch.nn as nn
import torch.nn.functional as F

class FireDetectionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(FireDetectionCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
  
        x = self.pool(F.relu(self.conv1(x)))
  
        x = self.pool(F.relu(self.conv2(x)))
        
        x = self.pool(F.relu(self.conv3(x)))
      
        x = x.view(-1, 128 * 28 * 28)
        
      
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":

    model = FireDetectionCNN()
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"Output shape: {output.shape}")
