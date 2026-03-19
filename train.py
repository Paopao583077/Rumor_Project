import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WeiboDataset
from model import RumorDetector


def main():
    # 1. 检查有没有显卡 (GPU)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在使用计算设备: {DEVICE}")

    # 2. 参数设置 (如果显存不够报错，就把 Batch Size 改小)
    BATCH_SIZE = 8
    LR = 2e-5
    EPOCHS = 3

    # 3. 准备数据
    print("📦 正在加载数据...")
    train_ds = WeiboDataset('./data/train.csv', mode='train')
    val_ds = WeiboDataset('./data/test.csv', mode='val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 初始化模型 (第一次运行会下载几百兆的模型文件，请耐心等待)
    print("🧠 正在初始化模型 (首次运行需下载预训练参数)...")
    model = RumorDetector().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 5. 循环训练
    print("▶️ 开始训练！")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, mask, images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch + 1} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"=== Epoch {epoch + 1} 完成 | 平均 Loss: {total_loss / len(train_loader):.4f} ===")

    # 6. 保存成果
    torch.save(model.state_dict(), 'best_model.pth')
    print("💾 训练完成，模型已保存为 best_model.pth")


if __name__ == '__main__':
    main()