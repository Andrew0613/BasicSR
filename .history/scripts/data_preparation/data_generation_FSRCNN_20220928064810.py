from PIL import Image
import numpy as np
import os
"""
按照FSRCNN论文中的说明构造数据集，参考data_aug.m文件
"""

def generate_train_img(dataset_path,output_path):
    """
    生成训练集
    :param dataset_path: 训练集路径
    :param output_path: 生成的训练集路径
    :return:
    """
    # 获取数据集路径
    img_path_list = os.listdir(dataset_path)

def data_augmentation(img):
    """
    数据增强
    :param img: 图片
    :return:
    """
    # 旋转
    img = img.rotate(90)
    # 翻转
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


if __name__=="__main__":
    dataset_path = "/ssd/home/uguser/pyd/dataset/image_SR/T91"
    output_path = "datasets"
    generate_train_img(dataset_path,output_path)