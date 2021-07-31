import os
from help_func.help_func import myUtil
from PIL import Image
target_list = ['/data4']
taget_ext = ['.png', '.jpg']

file_list = []
for ext in taget_ext:
    for target_path in target_list:
        file_list += myUtil.xgetFileList(target_path, ext=ext)

for file_path in file_list:
    img = Image.open(file_path)
    height, width = img.size
    if height % 8 or width % 8:
        new_height = height // 8 * 8
        new_width = width // 8 * 8
        crop_img = img.crop((0, 0, new_width, new_height))
        os.remove(file_path)
        crop_img.save(file_path)
        print('[INFO] Resize {} : {}x{} -> {}x{}'.format(file_path, height, width, new_height, new_width))


