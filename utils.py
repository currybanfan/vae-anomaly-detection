import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision.utils as vutils
from PIL import Image

def visualize_pixel_distribution(data_loader):
    """
    視覺化數據加載器中像素值的分佈情況，使用直方圖顯示。
    
    參數:
        data_loader: DataLoader
            PyTorch 的數據加載器，用於提供圖像數據。
    """
    all_pixels = []

    # 收集所有像素值
    for images, _ in data_loader:
        all_pixels.append(images)

    # 合併並展平像素值
    all_pixels = torch.cat(all_pixels).numpy().flatten()

    plt.hist(all_pixels, bins=50, color='blue', alpha=0.7)
    plt.title('Pixel Value Distribution (Train Set)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()


def show_8x4_images(data_loader, cmap='viridis'):
    """
    顯示數據加載器中32張圖像，排列為8x4網格。
    
    參數:
        data_loader: DataLoader
            PyTorch 的數據加載器，用於提供圖像數據。
        cmap: str
            用於顯示的顏色映射。
    """
    real_batch = next(iter(data_loader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(real_batch[0][:32], padding=2, normalize=True).cpu(), 
            (1, 2, 0)
        ), 
        cmap=cmap
    )
    plt.show()

def plot_training_validation_loss(train_losses, val_losses, ylim=None):
    """
    繪製訓練和驗證的損失曲線。
    
    參數:
        train_losses: list
            訓練損失值的列表。
        val_losses: list
            驗證損失值的列表。
        ylim: tuple 或 None
            損失曲線的 y 軸範圍。
    """
    plt.figure(figsize=(8, 6))
    plt.title('Training and Validation Loss')
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_reconstruction(model, loss_function, image_path, transform, inverse_transform, device):
    """
    可視化模型重建結果，並計算重建損失。
    
    參數:
        model: nn.Module
            訓練好的模型。
        loss_function: function
            用於計算損失的函數。
        image_path: str
            圖像的文件路徑。
        transform: function
            圖像預處理轉換。
        inverse_transform: function
            圖像的反轉換（將張量轉回圖像格式）。
        device: str
            運行模型的設備（如 'cpu' 或 'cuda'）。
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).to(device)
    img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 維度

    # 模型切換為評估模式
    model.eval()
    with torch.no_grad():
        recon, mu, log_var = model(img_tensor)
        
    test_loss = loss_function(recon, img_tensor, mu, log_var)
    print(f'loss: {test_loss.item()}')

    # 去掉 batch 維度
    original_img = img_tensor.squeeze(0).cpu()
    reconstructed_img = recon.squeeze(0).cpu()

    # 將重建圖像反正規化
    original_img = inverse_transform(original_img)
    reconstructed_img = inverse_transform(reconstructed_img)

    plt.figure(figsize=(8, 4))

    # 原始圖像
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis("off")

    # 重建圖像
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed_img)
    plt.axis("off")

    plt.show()

def visualize_gradients(model, gradient_type='both'):
    """
    視覺化模型的梯度分布，包括均值與標準差。
    
    參數:
        model: nn.Module
            欲分析的 PyTorch 模型
        gradient_type: str
            可選 'mean', 'std' 或 'both'，分別顯示梯度均值、標準差或兩者
    """
    names, grad_means, grad_stds = [], [], []

    # 收集梯度信息
    for name, param in model.named_parameters():
        if param.grad is not None:
            names.append(name)
            grad_means.append(param.grad.mean().item())
            grad_stds.append(param.grad.std().item())
    
    # 視覺化梯度均值
    if gradient_type in ('mean', 'both'):
        plt.figure(figsize=(12, 6))
        plt.bar(names, grad_means, color='blue', alpha=0.7, label='Gradient Mean')
        plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
        plt.title("Gradient Mean Across Model Parameters")
        plt.xlabel("Parameters")
        plt.ylabel("Gradient Mean")
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # 視覺化梯度標準差
    if gradient_type in ('std', 'both'):
        plt.figure(figsize=(12, 6))
        plt.bar(names, grad_stds, color='orange', alpha=0.7, label='Gradient Std')
        plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
        plt.title("Gradient Standard Deviation Across Model Parameters")
        plt.xlabel("Parameters")
        plt.ylabel("Gradient Std")
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_and_calculate_losses(good_losses, error_losses):
    """
    繪製正常和缺陷損失的分佈圖，並計算其平均值。
    
    參數:
        good_losses: list
            正常樣本的損失值。
        error_losses: list
            缺陷樣本的損失值。
    """
    plt.plot(error_losses, label='defective Loss')
    plt.scatter(range(len(error_losses)), error_losses, s=10)
    plt.plot(good_losses, label='good Loss')
    plt.scatter(range(len(good_losses)), good_losses, s=10)
    plt.xlabel('num')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    good_avg = sum(good_losses)/len(good_losses)
    defective_avg = sum(error_losses)/len(error_losses)

    print("good avg: ", good_avg)    
    print("defective avg: ", defective_avg)
