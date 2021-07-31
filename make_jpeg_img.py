from PIL import Image
from help_func.help_func import myUtil
import os
QUALITY_FACTOR = 20

hr_dir_list = ['/data4/DIV2K_train_HR', '/data4/DIV2K_valid_HR']
save_dir = '/data4'

if __name__ == '__main__':
    for hr_dir in hr_dir_list:
        lr_dir_path = os.path.join(save_dir,
                                   os.path.basename(hr_dir.replace('_HR',
                                                                   '_LR_Q' +
                                                                   str(QUALITY_FACTOR))))
        os.makedirs(lr_dir_path, exist_ok=True)
        file_list = myUtil.xgetFileList(hr_dir, '.png')
        for file_path in file_list:
            img = Image.open(file_path)
            newpath = os.path.join(lr_dir_path,
                                  os.path.basename(file_path)).replace('.png', '.jpg')
            img.save(newpath,
                     quality=QUALITY_FACTOR)
            print('[INFO] {} is saved'.format(newpath))


