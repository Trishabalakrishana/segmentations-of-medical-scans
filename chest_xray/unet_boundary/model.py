#model.py

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# ===============================
# Double Convolution Block
# ===============================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# ===============================
# U-Net Model
# ===============================
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling path (Decoder)
        for feature in reversed(features):
            # Upsampling layer
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            # Double conv after concatenation (feature from skip + feature from upsampling)
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder (Downsampling)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # Decoder (Upsampling)
        for idx in range(0, len(self.ups), 2):
            # Upsampling
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Handle size mismatch by cropping or padding
            if x.shape != skip_connection.shape:
                # Get the minimum dimensions
                min_h = min(x.shape[2], skip_connection.shape[2])
                min_w = min(x.shape[3], skip_connection.shape[3])
                
                # Crop both tensors to the same size
                x = x[:, :, :min_h, :min_w]
                skip_connection = skip_connection[:, :, :min_h, :min_w]

            # Concatenate skip connection with upsampled feature map
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # Apply double convolution
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

# ===============================
# Alternative U-Net with better size handling
# ===============================
class UNET_V2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET_V2, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling path (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling path (Decoder)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Better size handling using interpolation
            if x.shape[2:] != skip_connection.shape[2:]:
                x = TF.resize(x, size=skip_connection.shape[2:], antialias=False)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)

# ===============================
# Testing both models
# ===============================
def test():
    print("Testing UNET model...")
    try:
        x = torch.randn((3, 3, 160, 160))  # Batch of 3 RGB images
        model = UNET(in_channels=3, out_channels=1)
        preds = model(x)
        print(f"✅ UNET - Input shape: {x.shape}")
        print(f"✅ UNET - Output shape: {preds.shape}")
        assert preds.shape == (3, 1, 160, 160), f"Expected (3, 1, 160, 160), got {preds.shape}"
        print("✅ UNET test passed!")
    except Exception as e:
        print(f"❌ UNET test failed: {e}")

    print("\nTesting UNET_V2 model...")
    try:
        x = torch.randn((3, 3, 160, 160))
        model_v2 = UNET_V2(in_channels=3, out_channels=1)
        preds_v2 = model_v2(x)
        print(f"✅ UNET_V2 - Input shape: {x.shape}")
        print(f"✅ UNET_V2 - Output shape: {preds_v2.shape}")
        assert preds_v2.shape == (3, 1, 160, 160), f"Expected (3, 1, 160, 160), got {preds_v2.shape}"
        print("✅ UNET_V2 test passed!")
    except Exception as e:
        print(f"❌ UNET_V2 test failed: {e}")

    # Test with different input sizes
    print("\nTesting with different input sizes...")
    test_sizes = [(1, 3, 128, 128), (2, 3, 256, 256), (1, 3, 224, 224)]
    
    for batch_size, channels, height, width in test_sizes:
        try:
            x_test = torch.randn((batch_size, channels, height, width))
            model_test = UNET(in_channels=channels, out_channels=1)
            pred_test = model_test(x_test)
            print(f"✅ Input: {x_test.shape} -> Output: {pred_test.shape}")
        except Exception as e:
            print(f"❌ Failed for input size {(batch_size, channels, height, width)}: {e}")

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary():
    """Print model summary"""
    model = UNET(in_channels=3, out_channels=1)
    total_params = count_parameters(model)
    print(f"\n📊 Model Summary:")
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")

if __name__ == "__main__":
    test()
    model_summary()