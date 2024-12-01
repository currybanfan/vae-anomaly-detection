from dataset import TestFolderDataset
from torch.utils.data import DataLoader
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

def calculate_losses_for_category(folder_path, model, loss_function, transform, device):
    """
    計算資料夾中每張圖像的重建損失，並按標籤分類。
    
    參數:
        folder_path: str
            測試資料夾路徑。
        model: nn.Module
            預訓練的 VAE 模型。
        loss_function: function
            用於計算損失的函數。
        transform: callable
            對圖像應用的轉換。
        device: str
            設備名稱（如 'cpu' 或 'cuda'）。
    
    返回:
        tuple: (good_losses, error_losses)
    """
    test_dataset = TestFolderDataset(folder_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    good_losses = []
    error_losses = []


    model.eval() # 設置模型為評估模式
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            recon, mu, logvar = model(images)
            loss = loss_function(recon, images, mu, logvar)
            
            # 根據標籤分類損失
            if labels.item() == 0:  # good
                good_losses.append(loss.item())
            elif labels.item() == 1:  # error
                error_losses.append(loss.item())
    
    return good_losses, error_losses


def calculate_metrics(folder_path, model, loss_function, transform, threshold, device):
    """
    計算測試資料集的準確率、F1 分數和混淆矩陣。
    
    參數:
        folder_path: str
            測試資料夾路徑。
        model: nn.Module
            預訓練的 VAE 模型。
        loss_function: function
            用於計算損失的函數。
        transform: callable
            對圖像應用的轉換。
        threshold: float
            判斷 'good' 與 'error' 的損失閾值。
        device: str
            設備名稱（如 'cpu' 或 'cuda'）。
    
    返回:
        tuple: (accuracy, f1_score)
    """
    total = 0  # 總樣本數
    correct = 0  # 正確分類數
    all_labels = []  # 儲存所有真實標籤
    all_preds = []   # 儲存所有預測標籤

    test_dataset = TestFolderDataset(folder_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向傳播
            recon, mu, logvar = model(images)
            loss = loss_function(recon, images, mu, logvar).item()

            # 預測結果: 根據 loss 判斷是正常品還是瑕疵品
            pred = 0 if loss < threshold else 1  # 預測的標籤

            # 記錄真實標籤與預測標籤
            all_labels.append(labels.item())
            all_preds.append(pred)

            # 計算正確數量
            if pred == labels.item():
                correct += 1

            total += 1

    accuracy = correct / total if total > 0 else 0.0

    # 計算混淆矩陣
    cm = confusion_matrix(all_labels, all_preds)
    
    # 計算 F1-Score
    f1 = f1_score(all_labels, all_preds)

    # 顯示混淆矩陣
    plot_confusion_matrix(cm)

    return accuracy, f1

def plot_confusion_matrix(cm):
    """
    繪製混淆矩陣。
    
    參數:
        cm: ndarray
            混淆矩陣。
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Good', 'Error'], yticklabels=['Good', 'Error'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
