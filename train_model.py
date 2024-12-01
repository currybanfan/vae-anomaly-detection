import torch
import torch.nn as nn


def vae_loss(recon_x, x, mu, logvar):
    """
    計算變分自編碼器（VAE）的損失函數，包括重建損失和 KL 散度。
    
    參數:
        recon_x: Tensor
            模型重建的輸出圖像。
        x: Tensor
            原始輸入圖像。
        mu: Tensor
            潛在空間分布的均值。
        logvar: Tensor
            潛在空間分布的對數方差。
    
    返回:
        Tensor: 重建損失與 KL 散度的加總。
    """

    # 重建損失 (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL 散度，用於正則化潛在空間
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_divergence


def train_vae(model, train_loader, val_loader, optimizer, scheduler, loss_function, epochs=50, device='cpu'):
    """
    訓練變分自編碼器（VAE）模型。
    
    參數:
        model: nn.Module
            欲訓練的 VAE 模型。
        train_loader: DataLoader
            訓練集的數據加載器。
        val_loader: DataLoader
            驗證集的數據加載器。
        optimizer: Optimizer
            優化器（如 Adam）。
        scheduler: Scheduler
            用於調整學習率的調度器。
        loss_function: function
            損失函數（如 vae_loss）。
        epochs: int
            訓練的總輪數。
        device: str
            訓練運行的設備（如 'cpu' 或 'cuda'）。
    
    返回:
        tuple: 訓練損失列表和驗證損失列表。
    """
    train_losses = []  # 儲存每個 epoch 的訓練損失
    val_losses = []  # 儲存每個 epoch 的驗證損失

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()

            # 前向傳播
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        # 計算平均訓練損失
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0

        # 驗證集迭代
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                recon_x, mu, logvar = model(x)  # 編碼和解碼
                loss = loss_function(recon_x, x, mu, logvar)  # 計算損失
                   
                # 累加驗證損失
                val_loss += loss.item()
            
        # 計算平均驗證損失
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # 更新學習率
        scheduler.step(val_loss)

        # 每個 epoch 輸出損失
        print(f"Epoch [{epoch+1}/{epochs}]: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses