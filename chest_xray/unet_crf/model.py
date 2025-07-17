import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ===============================
# Attention Gate Module
# ===============================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ===============================
# Squeeze-and-Excitation Block
# ===============================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# ===============================
# Enhanced Double Convolution Block
# ===============================
class EnhancedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=True, dropout_rate=0.1):
        super(EnhancedDoubleConv, self).__init__()
        self.use_se = use_se
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        if use_se:
            self.se = SEBlock(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        
        if self.use_se:
            out = self.se(out)
            
        # Add residual connection
        if self.residual is not None:
            residual = self.residual(residual)
        
        if residual.shape == out.shape:
            out = out + residual
            
        return out

# ===============================
# ASPP (Atrous Spatial Pyramid Pooling) Module
# ===============================
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        
        # 1x1 conv
        conv1 = self.conv1(x)
        
        # Atrous convolutions
        atrous_outs = []
        for atrous_conv in self.atrous_convs:
            atrous_outs.append(atrous_conv(x))
        
        # Global pooling
        global_pool = self.global_pool(x)
        global_pool = F.interpolate(global_pool, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate all features
        out = torch.cat([conv1] + atrous_outs + [global_pool], dim=1)
        out = self.project(out)
        
        return out

# ===============================
# Enhanced U-Net with Attention and Deep Supervision
# ===============================
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], 
                 use_attention=True, use_deep_supervision=True, use_aspp=True):
        super(UNET, self).__init__()
        
        self.use_attention = use_attention
        self.use_deep_supervision = use_deep_supervision
        self.use_aspp = use_aspp
        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Attention gates
        if use_attention:
            self.attention_gates = nn.ModuleList()

        # Deep supervision outputs
        if use_deep_supervision:
            self.deep_outputs = nn.ModuleList()

        # Downsampling path (Encoder)
        for feature in features:
            self.downs.append(EnhancedDoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck with ASPP
        if use_aspp:
            self.bottleneck = nn.Sequential(
                EnhancedDoubleConv(features[-1], features[-1] * 2),
                ASPP(features[-1] * 2, features[-1] * 2)
            )
        else:
            self.bottleneck = EnhancedDoubleConv(features[-1], features[-1] * 2)

        # Upsampling path (Decoder)
        for i, feature in enumerate(reversed(features)):
            # Upsampling layer
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            
            # Attention gate
            if use_attention:
                self.attention_gates.append(AttentionGate(feature, feature, feature // 2))
            
            # Double conv after concatenation
            self.ups.append(EnhancedDoubleConv(feature * 2, feature))
            
            # Deep supervision output
            if use_deep_supervision and i < len(features) - 1:
                self.deep_outputs.append(nn.Conv2d(feature, out_channels, kernel_size=1))

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip_connections = []
        deep_outputs = []
        input_size = x.shape[-2:]

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

            # Apply attention gate
            if self.use_attention:
                skip_connection = self.attention_gates[idx // 2](x, skip_connection)

            # Handle size mismatch
            if x.shape[-2:] != skip_connection.shape[-2:]:
                x = F.interpolate(x, size=skip_connection.shape[-2:], mode='bilinear', align_corners=False)

            # Concatenate skip connection with upsampled feature map
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            # Apply double convolution
            x = self.ups[idx + 1](concat_skip)
            
            # Deep supervision
            if self.use_deep_supervision and idx // 2 < len(self.deep_outputs):
                deep_out = self.deep_outputs[idx // 2](x)
                deep_out = F.interpolate(deep_out, size=input_size, mode='bilinear', align_corners=False)
                deep_outputs.append(deep_out)

        # Final output
        main_output = self.final_conv(x)
        
        if self.use_deep_supervision and self.training:
            return main_output, deep_outputs
        else:
            return main_output

# ===============================
# Multi-Scale U-Net
# ===============================
class MultiScaleUNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(MultiScaleUNET, self).__init__()
        
        # Three scales of U-Net
        self.unet_full = UNET(in_channels, out_channels, features)
        self.unet_half = UNET(in_channels, out_channels, [f//2 for f in features])
        self.unet_quarter = UNET(in_channels, out_channels, [f//4 for f in features])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Full resolution
        out_full = self.unet_full(x)
        if isinstance(out_full, tuple):
            out_full = out_full[0]
        
        # Half resolution
        x_half = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        out_half = self.unet_half(x_half)
        if isinstance(out_half, tuple):
            out_half = out_half[0]
        out_half = F.interpolate(out_half, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Quarter resolution
        x_quarter = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        out_quarter = self.unet_quarter(x_quarter)
        if isinstance(out_quarter, tuple):
            out_quarter = out_quarter[0]
        out_quarter = F.interpolate(out_quarter, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Fuse all scales
        fused = torch.cat([out_full, out_half, out_quarter], dim=1)
        final_output = self.fusion(fused)
        
        return final_output

# ===============================
# Testing Functions
# ===============================
def test_models():
    print("Testing Enhanced U-Net models...")
    
    # Test Enhanced U-Net
    print("\n1. Testing EnhancedUNET...")
    try:
        x = torch.randn((2, 3, 256, 256))
        model = UNET(in_channels=3, out_channels=1)
        model.eval()
        
        with torch.no_grad():
            preds = model(x)
        
        print(f"✅ EnhancedUNET - Input: {x.shape}, Output: {preds.shape}")
        
        # Test training mode with deep supervision
        model.train()
        preds = model(x)
        if isinstance(preds, tuple):
            main_out, deep_outs = preds
            print(f"✅ Deep supervision - Main: {main_out.shape}, Deep outputs: {len(deep_outs)}")
        
    except Exception as e:
        print(f"❌ EnhancedUNET test failed: {e}")
    
    # Test Multi-Scale U-Net
    print("\n2. Testing MultiScaleUNET...")
    try:
        x = torch.randn((2, 3, 256, 256))
        model = MultiScaleUNET(in_channels=3, out_channels=1)
        model.eval()
        
        with torch.no_grad():
            preds = model(x)
        
        print(f"✅ MultiScaleUNET - Input: {x.shape}, Output: {preds.shape}")
        
    except Exception as e:
        print(f"❌ MultiScaleUNET test failed: {e}")

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_comparison():
    """Compare different model variants"""
    print("\n📊 Model Comparison:")
    print("-" * 60)
    
    models = {
        'EnhancedUNET (Full)': UNET(3, 1, use_attention=True, use_deep_supervision=True),
        'EnhancedUNET (No Attention)': UNET(3, 1, use_attention=False, use_deep_supervision=True),
        'EnhancedUNET (No Deep Sup)': UNET(3, 1, use_attention=True, use_deep_supervision=False),
        'MultiScaleUNET': MultiScaleUNET(3, 1),
    }
    
    for name, model in models.items():
        params = count_parameters(model)
        size_mb = params * 4 / 1024 / 1024
        print(f"{name:<25}: {params:>8,} params ({size_mb:.1f} MB)")

if __name__ == "__main__":
    test_models()
    model_comparison()