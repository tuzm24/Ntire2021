from PIL import Image
import cv2
import os
from help_func import myUtil

def pngfileOpenAndSaveByJPG(file_path, obj_folder, quality_factor):
    image = Image.open(file_path)
    rgb_im = image.convert('RGB')
    filename , _ = os.path.splitext(file_path)
    rgb_im.save(os.path.join(obj_folder, os.path.basename(filename) + '.jpg'), quality=quality_factor)
    return


def pngfileOpenAndSaveByBicubicInter(filepath, obj_folder, resize_factor):
    img = cv2.imread(filepath)
    save_img = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(obj_folder, os.path.basename(filepath)), save_img)
    return

def getJPGData(dir = './div2k/ntire_deblur_track1', quality_factor=25):
    dirs = myUtil.getDirlist(dir)
    for d in dirs:
        obj_dir = d + '_jpeg'
        obj_dir.replace('_sharp', '_blur')
        os.makedirs(os.path.join(obj_dir, '0'), exist_ok=True)
        file_list = myUtil.getFileList(d, ext='.png')
        for file in file_list:
            pngfileOpenAndSaveByJPG(file, obj_dir, quality_factor)
    return


def getCubicData(dir = './div2k/ntire_deblur_track2', resize_factor=1/4):
    dirs = myUtil.getDirlist(dir)
    for d in dirs:
        obj_dir = d + '_bicubic'
        obj_dir.replace('_sharp', '_blur')
        os.makedirs(os.path.join(obj_dir, '0'), exist_ok=True)
        file_list = myUtil.getFileList(d, ext='.png')
        for file in file_list:
            pngfileOpenAndSaveByBicubicInter(file, obj_dir, resize_factor)
    return

getCubicData()