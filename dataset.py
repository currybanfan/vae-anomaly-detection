from torch.utils.data import Dataset
import os
from PIL import Image

class SingleFolderDataset(Dataset):
    """
    參數:
        folder_path (str): 圖像所在的資料夾路徑。
        transform (callable, optional): 對圖像應用的轉換。
    """
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        參數:
            idx (int): 要取的圖像索引。
        
        Returns:
            tuple: (圖像, 標籤)，標籤固定為 0。
        """
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # 單類資料集標籤固定為 0
    

class TestFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        參數:
            folder_path (str): 資料夾路徑，包含 'good' 和 'error' 兩個子資料夾。
            transform (callable, optional): 對圖像應用的轉換。
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []  # 儲存所有圖像的路徑
        self.labels = []  # 儲存對應的標籤

        for label, subfolder in enumerate(['good', 'error']):  # 'good' 標籤為 0, 'error' 標籤為 1
            label_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(label_path):  # 檢查是否為有效資料夾
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    if os.path.isfile(img_path):  # 確保是文件
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        參數:
            idx (int): 要取的圖像索引。
        
        Returns:
            tuple: (圖像, 標籤)，標籤分別為 0（good）或 1（error）。
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")  # 確保圖片是 RGB 模式
        if self.transform:
            image = self.transform(image)
        return image, label  # 返回圖像與其對應的標籤