import torch
import torch.nn as nn
import torch.nn.functional as F

class DrivingPolicyNet(nn.Module):
    def __init__(self):
        super(DrivingPolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(15, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)

        self.move = nn.Linear(256, 3)
        self.turn = nn.Linear(256, 3)

    def forward(self, x):
        # x: (B, 15, 100, 200) from stacked grayscale images
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))

        x = self.pool(x).view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.move(x), self.turn(x)

# Run test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DrivingPolicyNet().to(device)
print(device)
dummy_input = torch.randn(1, 15, 100, 200).to(device)  # simulate stacked grayscale frames
with torch.no_grad():
    move_logits, turn_logits = model(dummy_input)

print("Move logits:", move_logits)
print("Turn logits:", turn_logits)