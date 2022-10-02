from PIL import Image
import numpy as np
import os
"""
按照FSRCNN论文中的说明构造数据集
首先要对数据集进行数据增强，增强的方法可以参考data_aug.m文件
然后对增强后的数据集进行采样，采样方法可以参考generate_training.m文件
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
    We augment the data in two ways. 1) Scaling: each image is downscaled with the factor 0.9, 0,8, 0.7 and 0.6. 2) Rotation: each image is rotated with the degree of 90, 180 and 270.
    :param img: img
    :return: list
    """
    for scale in range(6,10):
        for roatation in [0,90,180,270]:
            img = img.resize((img.size[0]//scale,img.size[1]//scale),Image.BICUBIC)
            img = img.rotate(roatation)
            yield img

if __name__=="__main__":
    dataset_path = "/ssd/home/uguser/pyd/dataset/image_SR/T91"
    output_path = "datasets"
    generate_train_img(dataset_path,output_path)