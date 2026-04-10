import torch
import torch.nn as nn


# ===================== Residual Block Definition =====================
class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


# ===================== U-LRM Net Model Definition =====================
class U_LRMNet(nn.Module):
    """U-shaped Lightweight Residual Multi-scale  Network (U-LRMNet)"""

    def __init__(self, input_channels=1, lstm_hidden=128, lstm_layers=2, num_classes=1):
        super(U_LRMNet, self).__init__()

        # Input Layer
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Residual Blocks - Encoder
        self.res_block1 = ResidualBlock(64, 64)
        self.res_block2 = ResidualBlock(64, 128, stride=2)
        self.res_block3 = ResidualBlock(128, 256, stride=2)
        self.res_block4 = ResidualBlock(256, 512, stride=2)

        # Pooling Layer
        self.pool_layer = nn.Sequential(
            ResidualBlock(512, 512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # LSTM Layer
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.3 if lstm_layers > 1 else 0
        )

        # Skip Connection
        self.skip_fc = nn.Linear(512, lstm_hidden)

        # Pooling after LSTM
        self.pool_fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Residual Blocks - Decoder
        self.decode_fc1 = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.decode_fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.decode_fc3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )

        # Skip to Output
        self.skip_to_output = nn.Linear(lstm_hidden, 64)

        # Output Layer
        self.output_layer = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Input layer
        x = self.input_conv(x)

        # Residual encoding
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)

        # Pooling
        x = self.pool_layer(x)
        x = x.view(batch_size, -1)  # [batch, 512]

        # LSTM processing
        lstm_input = x.unsqueeze(1)  # [batch, 1, 512]
        lstm_out, _ = self.lstm(lstm_input)  # [batch, 1, lstm_hidden]
        lstm_out = lstm_out.squeeze(1)  # [batch, lstm_hidden]

        # Skip connection (from pooling to LSTM output)
        skip = self.skip_fc(x)  # [batch, lstm_hidden]
        lstm_out = lstm_out + skip

        # Pooling layer
        x = self.pool_fc(lstm_out)

        # Residual decoding
        x = self.decode_fc1(x)
        x = self.decode_fc2(x)
        x = self.decode_fc3(x)

        # Skip connection to output
        skip_output = self.skip_to_output(lstm_out)
        x = x + skip_output

        # Output layer
        output = self.output_layer(x)

        return output
