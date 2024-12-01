import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class VAE_tanh(nn.Module):
    """
    使用 Tanh 激活函數的變分自編碼器（VAE）。
        - Encoder：將輸入圖像嵌入到潛在空間，輸出 mu 和 logvar。
        - Decoder：從潛在空間重建圖像，輸出範圍 [-1, 1]。
    """
    def __init__(self, input_dim=3, feature_dim=64, latent_dim=256, a=0.1):
        super(VAE_tanh, self).__init__()
        
        # 編碼器：多層卷積逐步減小空間分辨率
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, feature_dim, kernel_size=4, stride=2, padding=1),  # 256 -> 128
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(a),
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(a),
            nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(a),
            nn.Conv2d(feature_dim * 4, feature_dim * 8, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(a),
            nn.Conv2d(feature_dim * 8, feature_dim * 16, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.BatchNorm2d(feature_dim * 16),
            nn.LeakyReLU(a),

            nn.Flatten(),  # 展平
            nn.Linear(feature_dim * 16 * 8 * 8, 2048),
            nn.LeakyReLU(a),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 2 * latent_dim)  # mu 和 logvar
        )

        # 解碼器：將潛在變量映射回圖像
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LeakyReLU(a),
            nn.Dropout(p=0.2),
            nn.Linear(2048, feature_dim * 16 * 8 * 8),
            nn.LeakyReLU(a),
            nn.Unflatten(1, (feature_dim * 16, 8, 8)),  # 還原為張量形狀
            nn.ConvTranspose2d(feature_dim * 16, feature_dim * 8, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(a),
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(a),
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(a),
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(a),
            nn.ConvTranspose2d(feature_dim, input_dim, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh()  # 縮放到 [-1, 1]
        )

    def reparameterize(self, mu, logvar):
        # 使用 reparameterization trick 生成隨機變量 z
        std = torch.exp(0.5 * logvar)  # 標準差
        eps = torch.randn_like(std)  # 隨機噪聲
        return mu + eps * std  # 重新參數化

    def forward(self, x):
        # 前向傳播，包含編碼、重新參數化和解碼
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)  # 分割為 mu 和 logvar
        z = self.reparameterize(mu, logvar)  # 隨機採樣 z
        x_reconstructed = self.decoder(z)  # 重建圖像
        return x_reconstructed, mu, logvar
    


class VAE_sigmoid(nn.Module):
    """
    使用 Sigmoid 激活函數的變分自編碼器（VAE）。
        - Decoder 將輸出縮放到 [0, 1] 範圍。
    """
    def __init__(self, input_dim=3, feature_dim=64, latent_dim=256):
        # 編碼器和解碼器邏輯與 VAE_tanh 類似，主要差異在於激活函數為 Sigmoid
        super(VAE_sigmoid, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, feature_dim, kernel_size=4, stride=2, padding=1),  # 輸入 (3, 256, 256) -> 輸出 (feature_dim, 128, 128)
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),  # 輸出 (feature_dim * 2, 64, 64)
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(),
            nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1),  # 輸出 (feature_dim * 4, 32, 32)
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(),
            nn.Conv2d(feature_dim * 4, feature_dim * 8, kernel_size=4, stride=2, padding=1),  # 輸出 (feature_dim * 8, 16, 16)
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(),
            nn.Conv2d(feature_dim * 8, feature_dim * 16, kernel_size=4, stride=2, padding=1),  # 輸出 (feature_dim * 16, 8, 8)
            nn.BatchNorm2d(feature_dim * 16),
            nn.ReLU(),

            nn.Flatten(),  # 展平為一維
            nn.Linear(feature_dim * 16 * 8 * 8, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 2 * latent_dim)  # 輸出 (mu 和 logvar)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2048, feature_dim * 16 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (feature_dim * 16, 8, 8)),  # 還原維度

            # 還原到 (feature_dim * 8, 16, 16)
            nn.ConvTranspose2d(feature_dim * 16, feature_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(),

            # 還原到 (feature_dim * 4, 32, 32)
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(),

            # 還原到 (feature_dim * 2, 64, 64)
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(),

            # 還原到 (feature_dim, 128, 128)
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),

            nn.ConvTranspose2d(feature_dim, input_dim, kernel_size=4, stride=2, padding=1),  # 輸出 (3, 256, 256)
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # 標準差
        eps = torch.randn_like(std)  # 隨機噪聲
        return mu + eps * std  # 重新參數化

    def forward(self, x):
        # 編碼
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)  # 分割成 mu 和 logvar
        
        # 重新參數化
        z = self.reparameterize(mu, logvar)
        
        # 解碼
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
    
class VAE_ResNet(nn.Module):
    """
    使用 ResNet 結構作為編碼器的變分自編碼器。
        - 優點：ResNet 能有效提取深層特徵，改善梯度傳遞。
    """
    def __init__(self, input_dim=3, feature_dim=32, latent_dim=256):
        super(VAE_ResNet, self).__init__()

        # 使用 ResNetBlock 作為編碼器的基礎構建塊
        self.encoder = nn.Sequential(
            ResNetBlock(input_dim, feature_dim, stride=2),  # 256 -> 128
            ResNetBlock(feature_dim, feature_dim * 2, stride=2),  # 128 -> 64
            ResNetBlock(feature_dim * 2, feature_dim * 4, stride=2),  # 64 -> 32
            ResNetBlock(feature_dim * 4, feature_dim * 8, stride=2),  # 32 -> 16
            nn.Flatten(),
            nn.Linear(feature_dim * 8 * 16 * 16, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 2 * latent_dim)  # mu 和 logvar
        )

        self.decoder_input = nn.Linear(latent_dim, feature_dim * 8 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (feature_dim * 8, 16, 16)),  

            ResNetBlock(feature_dim * 8, feature_dim * 4, stride=1),  # 16 -> 16
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 4, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=4, stride=2, padding=1),  # 32 -> 64
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=4, stride=2, padding=1),  # 64 -> 128
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_dim, input_dim, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Sigmoid()  # 範圍壓縮到 [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(self.decoder_input(z))
        return x_reconstructed, mu, logvar
    

def initialize_weights(model):
    """
    初始化模型權重：
        - 卷積層使用 Kaiming 初始化。
        - 全連接層使用 Xavier 初始化。
        - BatchNorm 層初始化為 1 和 0。
    """
    for module in model.modules():
        # 初始化卷積層的權重和偏置
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nonlinearity = 'relu'
            a = 0  # 默認為 0 表示 ReLU
            
            # 檢查該層後的激活函數是否為 LeakyReLU
            if hasattr(module, 'activation'):
                if isinstance(module.activation, nn.LeakyReLU):
                    nonlinearity = 'leaky_relu'
                    a = module.activation.negative_slope  # 提取 LeakyReLU 的斜率
            
            # 根據激活函數選擇初始化
            init.kaiming_normal_(module.weight, nonlinearity=nonlinearity, a=a)
            
            if module.bias is not None:
                init.constant_(module.bias, 0)
        
        # 初始化全連接層的權重和偏置
        elif isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        
        # 初始化 BatchNorm 層
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)