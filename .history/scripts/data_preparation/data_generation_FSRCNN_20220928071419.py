from PIL import Image
import numpy as np
import os
"""
按照FSRCNN论文中的说明构造数据集
首先要对数据集进行数据增强，增强的方法可以参考data_aug.m文件
然后对增强后的数据集进行采样，采样方法可以参考generate_training.m文件
"""

def generate_train_img(dataset_path,output_path,scaling_factor=3):
    """
    生成训练集
    :param dataset_path: 训练集路径
    :param output_path: 生成的训练集路径
    :return:
    """
    # 获取数据集路径
    img_path_list = os.listdir(dataset_path)
    img_list = []
    #数据增强
    #We augment the data in two ways. 1) Scaling: each image is downscaled with the factor 0.9, 0,8, 0.7 and 0.6. 2) Rotation: each image is rotated with the degree of 90, 180 and 270.
    for img_path in img_path_list:
        img = Image.open(os.path.join(dataset_path,img_path))
        for scale in range(6,10):
            for roatation in [0,90,180,270]:
                new_img = img.resize((img.size[0]*scale//10,img.size[1]*scale//10),Image.BICUBIC)
                new_img = img.rotate(roatation)
                img_list.append(new_img)
    #采样
    #To prepare the training data, we first downsample the original training images by the desired scaling factor n to form the LR images. Then we crop the LR training images into a set of fsub × fsub -pixel sub-images with a stride k. The corresponding HR sub-images (with size (n*fsub)^2) are also cropped from the ground truth images. These LR/HR sub-image pairs are the primary training data.



if __name__=="__main__":
    dataset_path = "/ssd/home/uguser/pyd/dataset/image_SR/T91"
    output_path = "datasets"
    generate_train_img(dataset_path,output_path)