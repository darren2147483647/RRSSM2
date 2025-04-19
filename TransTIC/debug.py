import numpy as np
import torch
import os

'''
python demo/predict.py --config river_model/config.py --checkpoint river_model/latest.pth
python examples/segmentation.py -c config/segmentation.yaml -I
'''

def log_variable(variable, name, filename="debug_log.txt"):
    """
    將變數的詳細資訊記錄到文件中，包括變數名稱、資料型態、大小和值。

    Args:
        variable: 要記錄的變數。
        name: 變數的名稱。
        filename: 記錄檔案的名稱。
    """
    mode = "a" if os.path.exists(filename) else "w"
    with open(filename, mode) as f:
        f.write("-" * 40 + "\n")  # 變數之間的分隔符
        f.write(f"Variable Name: {name}\n")
        f.write(f"Data Type: {type(variable).__name__}\n")  # 使用 __name__ 取得型態名稱

        if isinstance(variable, np.ndarray):
            f.write(f"Size: {variable.shape}\n")
            f.write(f"Value:\n{variable}\n")
        elif isinstance(variable, torch.Tensor):
            f.write(f"Size: {variable.size()}\n")
            f.write(f"Value:\n{variable}\n")
        elif isinstance(variable, list):
            f.write(f"Size: {len(variable)}\n")
            f.write(f"Value: {variable}\n")
        elif isinstance(variable, str):
            f.write(f"Size: {len(variable)}\n")
            f.write(f"Value: {variable}\n")
        else:
            f.write(f"Size: N/A\n")  # 對於其他型態，大小顯示 N/A
            f.write(f"Value: {variable}\n")

        f.write("-" * 40 + "\n")  # 變數之間的分隔符

# 範例用法
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# tensor = torch.randn(2, 3)
# my_list = [1, "hello", 3.14]
# my_string = "This is a string."

# log_variable(arr, "my_array")
# log_variable(tensor, "my_tensor")
# log_variable(my_list, "my_list")
# log_variable(my_string, "my_string")
# log_variable(123, "my_integer")