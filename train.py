import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WeiboDataset
from model import RumorDetector
from baseline_models import TextOnlyModel, ImageOnlyModel, ConcatModel


def main():
    # 1. 检查有没有显卡 (GPU)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 正在使用计算设备: {DEVICE}")

    # 2. 参数设置
    BATCH_SIZE = 8
    LR = 2e-5
    EPOCHS = 3

    # --- 核心切换区：你需要跑哪个模型，就取消哪一行的注释，并给它起个名字 ---
    MODEL_TYPE = "text_only"
    model = TextOnlyModel().to(DEVICE)

    # MODEL_TYPE = "image_only"
    # model = ImageOnlyModel().to(DEVICE)

    # MODEL_TYPE = "concat"
    # model = ConcatModel().to(DEVICE)

    # MODEL_TYPE = "attention"
    # model = RumorDetector().to(DEVICE)
    # ------------------------------------------------------------------

    # 3. 准备数据
    print(f"📦 正在加载数据，准备训练模型: {MODEL_TYPE}...")
    train_ds = WeiboDataset('./data/train.csv')
    val_ds = WeiboDataset('./data/test.csv')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 初始化模型
    print(f"🧠 正在初始化 {MODEL_TYPE} 模型参数...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 5. 循环训练
    print(f"▶️ 开始训练 {MODEL_TYPE}！")
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
    save_name = f'best_model_{MODEL_TYPE}.pth'
    torch.save(model.state_dict(), save_name)
    print(f"💾 训练完成，模型已保存为 {save_name}")


if __name__ == '__main__':
    main()
