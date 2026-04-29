import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
import torchvision.transforms as transforms
import pandas as pd
import os


class WeiboDataset(Dataset):
    def __init__(self, csv_file, max_len=128):
        self.data = pd.read_csv(csv_file)
        # 加载 BERT 分词器（这一步会自动下载字典，约几十KB）
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.max_len = max_len

        # 定义图片“变身”规则
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 统一大小
            transforms.ToTensor(),  # 变数字
            # 归一化（让数据分布更标准，模型学得快）
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 1. 拿数据
        row = self.data.iloc[idx]
        text = str(row['text'])
        img_path = row['path']
        label = int(row['label'])

        # 2. 处理图片
        try:
            image = Image.open(img_path).convert('RGB')  # 确保是彩色图
            image = self.transform(image)
        except Exception as e:
            # 如果图片坏了，给一张全黑图凑数，防止报错中断训练
            image = torch.zeros(3, 224, 224)

        # 3. 处理文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',  # 不够长就补0
            truncation=True,  # 太长就截断
            return_attention_mask=True,
            return_tensors='pt',
        )

        # 4. 打包返回
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }