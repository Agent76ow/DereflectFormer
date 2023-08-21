# import os
# import shutil

# # 定义文件夹路径
# folder1 = r"D:\Code\DereflectFormer\datasets\REFLECT\CID\test\reflect"
# folder2 = r"D:\Code\DereflectFormer\datasets\REFLECT\train\ColoredDataset\cleaned\M"
# new_folder = r"D:\Code\DereflectFormer\datasets\REFLECT\CID\test\new reflect"

# # 获取第一个文件夹中的所有图片名称
# folder1_files = os.listdir(folder1)

# # 创建新的文件夹，如果它还不存在的话
# os.makedirs(new_folder, exist_ok=True)

# # 遍历第二个文件夹中的所有图片
# for filename in os.listdir(folder2):
#     # 获取图片名称的前缀
#     prefix = filename.split('_')[0]
#     # 检查是否有与该前缀相同的图片在第一个文件夹中
#     for file in folder1_files:
#         if file.startswith(prefix):
#             # 如果有，将图片复制到新的文件夹中，并更改其名称
#             shutil.copy(os.path.join(folder2, filename), os.path.join(new_folder, file))
#             break


import os
from PIL import Image

# 图片文件夹路径
folder_path = "D:\\Code\\DereflectFormer\\datasets\\REFLECT\\CID\\test\\reflect"

# 获取文件夹中的所有文件名
files = os.listdir(folder_path)

for file in files:
    # 检查文件是否是图片（这里只检查了.jpg和.png两种格式，如果有其他格式你需要自己添加）
    if file.endswith('.jpg') or file.endswith('.JPG'):
        # 获取图片的完整路径
        img_path = os.path.join(folder_path, file)
        
        # 打开并缩放图片
        img = Image.open(img_path)
        img_resized = img.resize((800, 600))
        
        # 保存缩放后的图片到原来的路径（这将覆盖原图片，如果你不想覆盖原图片，可以选择保存到其他路径）
        img_resized.save(img_path)
