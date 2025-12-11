import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(w):
    if isinstance(w, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(w.weight, 0.0, 0.02)
        if w.bias is not None:
            nn.init.zeros_(w.bias)
    elif isinstance(w, nn.Linear):
        nn.init.normal_(w.weight, 0.0, 0.02)
        if w.bias is not None:
            nn.init.zeros_(w.bias)
    elif isinstance(w, nn.BatchNorm2d):
        nn.init.normal_(w.weight, 1.0, 0.02)
        nn.init.zeros_(w.bias)

class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x*x, dim=1, keepdim=True) + self.eps)

class UpProjection(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.projection = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=1, stride=1, padding=0, bias=False
            ),
            PixelNorm()
        )

    def forward(self, x):
        return self.scale_factor * self.projection(x)

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        self.residual = UpProjection(in_channels=in_channels, out_channels=out_channels)
        self.norm = PixelNorm()
        self.act = nn.GELU()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv(x) + residual
        x = self.norm(x)
        x = self.act(x)
        return x

class Generator(nn.Module):
    def __init__(self, num_dims=100, num_classes=2, embed_dims=2):
        super().__init__()

        self.label_embed = nn.Sequential(
            nn.Embedding(num_embeddings=num_classes, embedding_dim=embed_dims),
        )

        # ( 1 x 1 x num_dims+embed_dims => 4 x 4 x 1024)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=num_dims+embed_dims, out_channels=1024,
                kernel_size=4, stride=1, padding=0, bias=False
            ),
            PixelNorm(),
            nn.GELU()
        )

        # ( 4 x 4 x 1024 => 8 x 8 x 512)
        self.block2 = GeneratorBlock(in_channels=1024, out_channels=512)

        # ( 8 x 8 x 512 => 16 x 16 x 256)
        self.block3 = GeneratorBlock(in_channels=512, out_channels=256)

        # ( 16 x 16 x 256 => 32 x 32 x 128)
        self.block4 = GeneratorBlock(in_channels=256, out_channels=128)

        # ( 32 x 32 x 128 => 64 x 64 x 3)
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=3,
                kernel_size=4, stride=2, padding=1, bias=True
            ),
            nn.Tanh()
        )

    def forward(self, noises, labels):
        labels = self.label_embed(labels)                                # [B] => [B, embed_dims]
        labels = labels.unsqueeze(2).unsqueeze(3)                        # [B, embed_dims, 1, 1]

        x = torch.cat([noises, labels], dim=1)                           # [B, num_dims+embed_dims, 1, 1]
        x = self.block1(x)                                               # [B, 1024, 4, 4]
        x = self.block2(x)                                               # [B, 512, 8, 8]
        x = self.block3(x)                                               # [B, 256, 16, 16]
        x = self.block4(x)                                               # [B, 128, 32, 32]
        x = self.block5(x)                                               # [B, 3, 64, 64]
        return x

class Discriminator(nn.Module):
    def __init__(self, num_classes=2, embed_dims=2):
        super().__init__()

        self.label_embed = nn.Sequential(
            nn.Embedding(num_embeddings=num_classes, embedding_dim=embed_dims)
        )

        # ( 64 x 64 x 3+embed_dims => 32 x 32 x 64)
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3+embed_dims, out_channels=64,
                kernel_size=4, stride=2, padding=1, bias=True
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # ( 32 x 32 x 64 => 16 x 16 x 128)
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # ( 16 x 16 x 128 => 8 x 8 x 256)
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # ( 8 x 8 x 256 => 4 x 4 x 512)
        self.block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # ( 4 x 4 x 512 => 1 x 1 x 1)
        self.block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1,
                kernel_size=4, stride=1, padding=0, bias=True
            )
        )

    def forward(self, images, labels):
        labels = self.label_embed(labels)                                # [B] => [B, embed_dims]
        labels = labels.unsqueeze(2).unsqueeze(3)                        # [B, embed_dims, 1, 1]
        labels = labels.expand(-1, -1, 64, 64)                            # [B, embed_dims, 1, 1] => [B, embed_dims, 64, 64]

        x = torch.cat([images, labels], dim=1)                           # [B, 3+embed_dims, 64, 64]
        x = self.block1(x)                                               # [B, 3+embed_dims, 64, 64] => [B, 64, 32, 32]
        x = self.block2(x)                                               # [B, 64, 32, 32] => [B, 128, 16, 16]
        x = self.block3(x)                                               # [B, 128, 16, 16] => [B, 256, 8, 8]
        x = self.block4(x)                                               # [B, 256, 8, 8] => [B, 512, 4, 4]
        x = self.block5(x)                                               # [B, 512, 4, 4] => [B, 1, 1, 1]
        return x