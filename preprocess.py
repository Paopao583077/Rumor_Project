import os
import pandas as pd


def parse_raw_txt(rumor_file, nonrumor_file, output_csv):
    """
    功能：读取乱糟糟的txt，整理成整齐的csv
    """
    print(f"正在生成 {output_csv} ...")
    data_rows = []

    def process_file(filepath, label_value):
        # 1. 检查文件是否存在
        if not os.path.exists(filepath):
            print(f"⚠️ 警告：找不到文件 {filepath}")
            return

        print(f"  -> 正在读取: {os.path.basename(filepath)}")

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        count = 0
        # 循环读取：步长为3，因为每3行是一条数据
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines): break

            # --- 提取图片 ---
            # 第2行包含图片链接，我们只取文件名
            image_line = lines[i + 1].strip()
            image_urls = image_line.split('|')

            target_image_name = None
            for url in image_urls:
                if url != 'null' and url.endswith('.jpg'):
                    target_image_name = url.split('/')[-1]  # 拿到 xxx.jpg
                    break

            # 如果这条微博没图，就跳过（因为我们做的是多模态研究，必须有图）
            if target_image_name is None:
                continue

                # 告诉代码，图片统一在 ./data/images/ 里面
            image_path = f"./data/images/{target_image_name}"

            # --- 提取文本 ---
            # 第3行是微博正文
            text_content = lines[i + 2].strip()
            if not text_content:
                text_content = "无文本"

            # 存入列表
            data_rows.append({
                'path': image_path,
                'text': text_content,
                'label': label_value
            })
            count += 1
        print(f"     提取成功: {count} 条")

    # 1 代表谣言，0 代表非谣言
    process_file(rumor_file, 1)
    process_file(nonrumor_file, 0)

    # 保存成表格
    if len(data_rows) > 0:
        df = pd.DataFrame(data_rows)
        # 打乱顺序（洗牌），防止模型死记硬背
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(output_csv, index=False)
        print(f"✅ 成功！已生成 {output_csv} (共 {len(df)} 条数据)\n")
    else:
        print("❌ 失败：没有提取到任何数据。\n")


if __name__ == '__main__':
    # 任务1：制作训练集表格
    parse_raw_txt(
        rumor_file='./data/tweets/train_rumor.txt',
        nonrumor_file='./data/tweets/train_nonrumor.txt',
        output_csv='./data/train.csv'
    )

    # 任务2：制作测试集表格
    parse_raw_txt(
        rumor_file='./data/tweets/test_rumor.txt',
        nonrumor_file='./data/tweets/test_nonrumor.txt',
        output_csv='./data/test.csv'
    )