import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from dataset import WeiboDataset
from model import RumorDetector
from baseline_models import TextOnlyModel, ImageOnlyModel, ConcatModel
import numpy as np


def evaluate():
    # 1. 设置
    BATCH_SIZE = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 开始评估，使用设备: {device}")

    # --- 核心切换区：评估哪个模型就改这里 ---
    MODEL_TYPE = "text_only"  # 可选: "text_only", "image_only", "concat", "attention"
    
    if MODEL_TYPE == "text_only":
        model = TextOnlyModel()
    elif MODEL_TYPE == "image_only":
        model = ImageOnlyModel()
    elif MODEL_TYPE == "concat":
        model = ConcatModel()
    else:
        model = RumorDetector()
        
    model_path = f'best_model_{MODEL_TYPE}.pth'
    # 兼容旧的文件名，如果不存在带后缀的，就尝试加载 best_model_attention.pth
    import os
    if not os.path.exists(model_path) and MODEL_TYPE == "attention":
        model_path = 'best_model_attention.pth'
        
    print(f"🔍 正在加载模型权重: {model_path}")
    # --------------------------------------

    # 2. 加载测试集
    test_ds = WeiboDataset('./data/test.csv')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"📦 测试集包含 {len(test_ds)} 条数据")

    # 3. 加载模型
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 4. 开始考试
    print("▶️ 正在进行批量预测，请稍候...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, mask, images)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if i % 10 == 0:
                print(f"   已处理 {i * BATCH_SIZE} 条...")

    # 5. 计算指标
    print("\n" + "=" * 40)
    print("🏆 最终评估报告")
    print("=" * 40)

    # 准确率
    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ 总体准确率 (Accuracy): {acc:.4f}")

    # 详细报告 (F1值, 召回率等)
    print("\n📊 详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['非谣言', '谣言'], digits=4))

    # 混淆矩阵
    print("\n🧩 混淆矩阵 (Confusion Matrix):")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"TP (谣言判对): {cm[1][1]}")
    print(f"TN (非谣言判对): {cm[0][0]}")
    print(f"FP (误判为谣言): {cm[0][1]}")
    print(f"FN (漏判谣言): {cm[1][0]}")
    print("=" * 40)


if __name__ == '__main__':
    evaluate()