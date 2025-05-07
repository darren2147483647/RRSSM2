import os
import matplotlib.pyplot as plt
from collections import defaultdict

# 所有檔案所在目錄
data_dir = 'rdresult'

# 用來儲存每個模型的 bpp 和 iou
rd_data = defaultdict(lambda: {'bpp': [], 'iou': []})

# 讀取所有 txt 檔
for filename in os.listdir(data_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r') as f:
            for line in f:
                path, bpp, iou = line.strip().split(',')
                bpp = float(bpp)
                iou = float(iou)

                # 根據路徑中 'ckptX' 分類（可改為你想要的標籤邏輯）
                model_key = '_'.join(os.path.basename(filename).split('_')[:-1])

                rd_data[model_key]['bpp'].append(bpp)
                rd_data[model_key]['iou'].append(iou)

# 繪圖
plt.figure()
for model_key, data in rd_data.items():
    # 確保按 bpp 排序（可選）
    sorted_data = sorted(zip(data['bpp'], data['iou']))
    bpp_sorted, iou_sorted = zip(*sorted_data)

    plt.plot(bpp_sorted, iou_sorted, marker='o', label=model_key)

plt.axhline(y=0.97635, color='gray', linestyle='--', label='Origin IoU avg: 0.97635') # origin iou avg: 0.97635
plt.xlabel('bpp (bit per pixel)')
plt.ylabel('IoU')
plt.title('Rate-Distortion Curve')
plt.grid(True)
plt.legend()
plt.savefig('rd_curve.png')  # 或 plt.show()

