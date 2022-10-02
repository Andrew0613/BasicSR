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
    img_list = []
    for img_path in img_path_list:
        img = Image.open(os.path.join(dataset_path,img_path))
        for scale in range(6,10):
            for roatation in [0,90,180,270]:
                new_img = img.resize((img.size[0]*scale//10,img.size[1]*scale//10),Image.BICUBIC)
                new_img = img.rotate(roatation)
                img_list.append(new_img)


if __name__=="__main__":
    dataset_path = "/ssd/home/uguser/pyd/dataset/image_SR/T91"
    output_path = "datasets"
    generate_train_img(dataset_path,output_path)