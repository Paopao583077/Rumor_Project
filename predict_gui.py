import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from PIL import Image, ImageTk
from transformers import BertTokenizer
import torchvision.transforms as transforms
import os
import io

# 从项目导入模型结构 (确保 model.py 存在)
from model import RumorDetector

# ==================== 核心配置 ====================
MODEL_PATH = 'best_model.pth'
IMAGE_SIZE = (200, 200)  # GUI中展示的图片大小

# 1. 初始化模型和设备 (只运行一次)
try:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    model = RumorDetector().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 图片预处理流水线
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 模型输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"✅ 模型已加载完毕，使用设备: {DEVICE}")
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ 模型加载失败！请检查 {MODEL_PATH} 是否存在。错误: {e}")


# =================================================

class RumorDetectorGUI:
    def __init__(self, master):
        self.master = master
        master.title("多模态谣言检测系统")
        master.resizable(False, False)

        self.image_path = None

        # --- 样式设置 ---
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Helvetica', 10), padding=5)
        style.configure('TLabel', font=('Helvetica', 10))

        # --- 布局框架 ---
        main_frame = ttk.Frame(master, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- 1. 文本输入 ---
        ttk.Label(main_frame, text="新闻文本输入:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.text_input = tk.Text(main_frame, height=5, width=60, font=('Helvetica', 10))
        self.text_input.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.text_input.insert(tk.END, "")  # 默认文本

        # --- 2. 图片上传 ---
        ttk.Label(main_frame, text="2. 上传配图:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.upload_button = ttk.Button(main_frame, text="选择图片文件...", command=self.select_image)
        self.upload_button.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)

        self.image_label = ttk.Label(main_frame, text="[无图片]", background="lightgrey", width=30, anchor="center")
        self.image_label.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

        # --- 3. 预测按钮 ---
        self.predict_button = ttk.Button(main_frame, text="开始多模态分析预测", command=self.predict,
                                         style='Accent.TButton')
        self.predict_button.grid(row=4, column=0, columnspan=2, sticky=tk.W + tk.E, padx=5, pady=10)

        # --- 4. 结果展示 ---
        ttk.Label(main_frame, text="3. 模型预测结果:").grid(row=5, column=0, sticky=tk.W, pady=5)

        # 结果主显示
        self.result_text = tk.StringVar()
        self.result_text.set("--- 等待预测 ---")
        self.result_label = ttk.Label(main_frame, textvariable=self.result_text, font=('Helvetica', 14, 'bold'),
                                      anchor="center")
        self.result_label.grid(row=6, column=0, columnspan=2, sticky=tk.W + tk.E, padx=5, pady=5)

        # 概率条展示
        self.prob_frame = ttk.Frame(main_frame)
        self.prob_frame.grid(row=7, column=0, columnspan=2, sticky=tk.W + tk.E, padx=5, pady=5)

        ttk.Label(self.prob_frame, text="谣言概率:").grid(row=0, column=0, sticky=tk.W)
        self.prob_rumor_var = tk.StringVar()
        ttk.Label(self.prob_frame, textvariable=self.prob_rumor_var).grid(row=0, column=1, sticky=tk.W, padx=10)
        self.prob_rumor_bar = ttk.Progressbar(self.prob_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.prob_rumor_bar.grid(row=0, column=2, sticky=tk.W + tk.E)

        ttk.Label(self.prob_frame, text="非谣言概率:").grid(row=1, column=0, sticky=tk.W)
        self.prob_nonrumor_var = tk.StringVar()
        ttk.Label(self.prob_frame, textvariable=self.prob_nonrumor_var).grid(row=1, column=1, sticky=tk.W, padx=10)
        self.prob_nonrumor_bar = ttk.Progressbar(self.prob_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.prob_nonrumor_bar.grid(row=1, column=2, sticky=tk.W + tk.E)

        # 5. 图片预览
        self.preview_label = ttk.Label(main_frame, text="图片预览")
        self.preview_label.grid(row=8, column=0, columnspan=2, pady=10)
        self.image_display = ttk.Label(main_frame, text="")
        self.image_display.grid(row=9, column=0, columnspan=2, pady=5)

    def select_image(self):
        """打开文件对话框选择图片"""
        f_types = [('Jpg Files', '*.jpg'), ('Png Files', '*.png')]
        self.image_path = filedialog.askopenfilename(filetypes=f_types)

        if self.image_path:
            self.image_label.config(text=os.path.basename(self.image_path))
            # 刷新图片预览
            try:
                img = Image.open(self.image_path)
                img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                self.tk_img = ImageTk.PhotoImage(img)
                self.image_display.config(image=self.tk_img)
            except Exception as e:
                self.image_display.config(text=f"预览失败: {e}")
                self.tk_img = None
        else:
            self.image_label.config(text="[无图片]")
            self.image_display.config(image='')
            self.tk_img = None

    def predict(self):
        """执行模型预测"""
        if not MODEL_LOADED:
            messagebox.showerror("错误", "模型未加载成功，请检查 best_model.pth 文件。")
            return

        text = self.text_input.get("1.0", tk.END).strip()

        if not text or not self.image_path:
            messagebox.showwarning("输入警告", "请确保文本框和图片都已选择。")
            return

        # 禁用按钮，显示加载状态
        self.predict_button.config(state=tk.DISABLED, text="分析中...")

        try:
            # 1. 处理图片 (用于模型输入)
            image_tensor = self._process_image_for_model(self.image_path)

            # 2. 处理文本
            encoding = tokenizer.encode_plus(
                text, add_special_tokens=True, max_length=128,
                padding='max_length', truncation=True, return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(DEVICE)
            attention_mask = encoding['attention_mask'].to(DEVICE)

            # 3. 预测
            with torch.no_grad():
                logits = model(input_ids, attention_mask, image_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)

                prob_non_rumor = probs[0].item()
                prob_rumor = probs[1].item()

                pred = 1 if prob_rumor > prob_non_rumor else 0

            # 4. 更新界面
            self._update_results(pred, prob_non_rumor, prob_rumor)

        except Exception as e:
            messagebox.showerror("预测失败", f"模型预测过程中发生错误: {e}")
            self.result_text.set("预测失败")
        finally:
            # 恢复按钮状态
            self.predict_button.config(state=tk.NORMAL, text="开始多模态分析预测")

    def _process_image_for_model(self, path):
        """内部方法：将图片处理成模型所需的 Tensor"""
        try:
            image = Image.open(path).convert('RGB')
            # 应用我们在全局定义的 transform (resize到224x224, 归一化等)
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            return image_tensor
        except Exception:
            # 如果图片损坏，返回一个全黑的 Tensor
            return torch.zeros(1, 3, 224, 224).to(DEVICE)

    def _update_results(self, pred, prob_non_rumor, prob_rumor):
        """更新结果显示区域"""

        if pred == 1:
            result_str = "🔴 判决结果：谣言 (RUMOR)"
            color = "#ff0000"  # 红色
        else:
            result_str = "🟢 判决结果：非谣言 (NON-RUMOR)"
            color = "#008000"  # 绿色

        # 更新主结果文本
        self.result_text.set(result_str)
        self.result_label.config(foreground=color)

        # 更新概率和进度条
        self.prob_rumor_var.set(f"{prob_rumor * 100:.2f}%")
        self.prob_nonrumor_var.set(f"{prob_non_rumor * 100:.2f}%")

        self.prob_rumor_bar['value'] = prob_rumor * 100
        self.prob_nonrumor_bar['value'] = prob_non_rumor * 100

        # 让非谣言的进度条颜色变成绿色
        self.prob_nonrumor_bar.config(style='green.Horizontal.TProgressbar')
        self.prob_rumor_bar.config(style='red.Horizontal.TProgressbar')

        # 自定义进度条颜色样式
        style = ttk.Style()
        style.configure("red.Horizontal.TProgressbar", foreground='red', background='red')
        style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')


if __name__ == '__main__':
    if MODEL_LOADED:
        root = tk.Tk()
        app = RumorDetectorGUI(root)
        root.mainloop()
    else:
        # 如果模型没加载成功，直接给出提示
        root = tk.Tk()
        root.title("错误")
        ttk.Label(root, text="模型加载失败，无法启动演示程序！", foreground="red").pack(padx=20, pady=20)
        root.mainloop()