from PIL import Image
import numpy as np
import os
import h5py
from tqdm import tqdm
"""
按照FSRCNN论文中的说明构造数据集
首先要对数据集进行数据增强，增强的方法可以参考data_aug.m文件
然后对增强后的数据集进行采样，采样方法可以参考generate_training.m文件
"""

def generate_train_img(dataset_path,output_path,scale=3,patch_size=11):
    """
    生成训练集
    :param dataset_path: 训练集路径
    :param output_path: 生成的训练集路径
    :return:
    """
    # 获取数据集路径
    img_path_list = sorted(os.listdir(dataset_path))
    img_list = []
    #数据增强
    #We augment the data in two ways. 1) Scaling: each image is downscaled with the factor 0.9, 0,8, 0.7 and 0.6. 2) Rotation: each image is rotated with the degree of 90, 180 and 270.
    for img_path in tqdm(img_path_list):
        img = Image.open(os.path.join(dataset_path,img_path))
        for scale in range(6,10):
            for roatation in [0,90,180,270]:
                new_img = img.resize((img.size[0]*scale//10,img.size[1]*scale//10),Image.BICUBIC)
                new_img = img.rotate(roatation)
                img_list.append(new_img)
    #采样
    #To prepare the training data, we first downsample the original training images by the desired scaling factor n to form the LR images. Then we crop the LR training images into a set of fsub × fsub -pixel sub-images with a stride k. The corresponding HR sub-images (with size (n*fsub)^2) are also cropped from the ground truth images. These LR/HR sub-image pairs are the primary training data.
    lr_patches = []
    hr_patches = []
    for hr in tqdm(img_list):
            hr_width = (hr.width // scale) * scale
            hr_height = (hr.height // scale) * scale
            hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
            lr = hr.resize((hr.width // scale, hr_height // scale), resample=Image.BICUBIC)
            hr = np.array(hr).astype(np.float32)
            lr = np.array(lr).astype(np.float32)
            # hr = rgb2YCbCr(hr)
            # lr = rgb2YCbCr(lr)

            for i in range(0, lr.shape[0] - patch_size + 1, scale):
                for j in range(0, lr.shape[1] - patch_size + 1, scale):
                    lr_patches.append(lr[i:i+patch_size, j:j+patch_size])
                    hr_patches.append(hr[i*scale:i*scale+patch_size*scale, j*scale:j*scale+patch_size*scale])
    #保存数据集
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    #删除文件夹
    if os.path.exists(os.path.join(output_path,'lr%s'%(scale))):
        os.system('rm -rf %s'%(os.path.join(output_path,'lr%s'%(scale))))
    if os.path.exists(os.path.join(output_path,'hr%s'%(scale))):
        os.system('rm -rf %s'%(os.path.join(output_path,'hr%s'%(scale))))
    #创建文件夹
    if not os.path.exists(os.path.join(output_path,'lr%s'%(scale))):
        os.makedirs(os.path.join(output_path,'lr%s'%(scale)))
    if not os.path.exists(os.path.join(output_path,'hr%s'%(scale))):
        os.makedirs(os.path.join(output_path,'hr%s'%(scale)))
    for i in tqdm(range(len(lr_patches))):
        #使用image库保存图片
        lr_img = Image.fromarray(lr_patches[i].astype(np.uint8))
        hr_img = Image.fromarray(hr_patches[i].astype(np.uint8))
        lr_img.save(os.path.join(output_path,'lr%s'%(scale),str(i)+'.png'))
        hr_img.save(os.path.join(output_path,'hr%s'%(scale),str(i)+'.png'))
    print('Done')
def rgb2YCbCr(img):
    """
    RGB转YCbCr
    :param img:
    :return:
    """
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    Y = 16 + (65.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2])/256
    Cb = 128 + (-37.945 * img[:, :, 0]  - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2])/256
    Cr = 128 + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2])/256
    YCbCr = np.zeros(img.shape)
    YCbCr[:, :, 0] = Y
    YCbCr[:, :, 1] = Cb
    YCbCr[:, :, 2] = Cr
    return YCbCr


if __name__=="__main__":
    dataset_path = "/ssd/home/uguser/pyd/dataset/image_SR/T91"
    output_path = "datasets"
    generate_train_img(dataset_path,output_path)