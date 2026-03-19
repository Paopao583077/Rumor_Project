---
description: 初始化谣言检测项目环境
---

# 项目初始化工作流

此工作流将帮助您设置谣言检测项目的开发环境。

## 步骤

### 1. 检查 Python 环境
确认 Python 版本（建议 3.8+）：
```bash
python --version
```

### 2. 激活虚拟环境
如果 `.venv` 目录已存在，激活它：
```bash
.venv\Scripts\activate
```

如果虚拟环境不存在，创建新的：
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. 安装项目依赖
安装必要的 Python 包：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas pillow scikit-learn matplotlib tqdm
pip install tkinter
```

### 4. 验证 BERT 模型
检查 BERT 中文模型缓存（首次使用会自动下载）：
```python
python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-chinese')"
```

### 5. 检查数据目录
确认 `data` 目录存在且包含必要文件：
```bash
ls data
```

应包含：
- `train.csv` - 训练数据
- `test.csv` - 测试数据
- 图片文件夹

### 6. 验证代码文件
确认所有核心文件存在：
- `dataset.py` - 数据集加载
- `model.py` - 模型定义
- `train.py` - 训练脚本
- `evaluate.py` - 评估脚本
- `preprocess.py` - 数据预处理
- `predict_gui.py` - GUI 预测界面

### 7. 运行快速测试
测试数据加载是否正常：
```bash
python -c "from dataset import WeiboDataset; print('数据集加载成功！')"
```

## 完成

环境初始化完成！您现在可以：
- 运行 `python train.py` 开始训练
- 运行 `python evaluate.py` 评估模型
- 运行 `python predict_gui.py` 启动 GUI 界面
