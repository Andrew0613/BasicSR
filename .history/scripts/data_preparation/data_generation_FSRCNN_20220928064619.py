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
    We augment the data in two ways. 1) Scaling: each image is downscaled with the factor 0.9, 0,8, 0.7 and 0.6. 2) Rotation: each image is rotated with the degree of 90, 180 and 270.

    """
    # 获取数据集路径
    img_path_list = os.listdir(dataset_path)
    for img_path in img_path_list:
        img = Image.open(os.path.join(dataset_path,img_path))
        img = np.array(img)





if __name__=="__main__":
    dataset_path = "/ssd/home/uguser/pyd/dataset/image_SR/T91"
    output_path = "datasets"
    generate_train_img(dataset_path,output_path)