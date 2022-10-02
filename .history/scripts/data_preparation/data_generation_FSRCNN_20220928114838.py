from operator import is_
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

def generate_train_img(dataset_path,output_path,up_scale=3,patch_size=11,is_residual=False):
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
    for label in tqdm(img_list):
            label_width = (label.width // up_scale) * up_scale
            label_height = (label.height // up_scale) * up_scale
            label = label.resize((label_width, label_height), resample=Image.BICUBIC)
            data = label.resize((label.width // up_scale, label_height // up_scale), resample=Image.BICUBIC)
            label = label.convert('YCbCr')
            data = data.convert('YCbCr')
            label = np.array(label).astype(np.float32)
            data = np.array(data).astype(np.float32)
            for i in range(0, data.shape[0] - patch_size + 1, up_scale):
                for j in range(0, data.shape[1] - patch_size + 1, up_scale):
                    lr_patches.append(data[i:i+patch_size, j:j+patch_size])
                    hr_patches.append(label[i*up_scale:i*up_scale+patch_size*up_scale, j*up_scale:j*up_scale+patch_size*up_scale])
    #保存数据集
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    #这种方式保存速度较快，但是框架里没有相应的读取方式
    # if is_residual:
    #     h5_file = h5py.File(os.path.join(output_path,"train_x{}_residual".format(up_scale)), 'w')
    # else:
    #      h5_file = h5py.File(os.path.join(output_path,"train_x{}".format(up_scale)), 'w')
    # h5_file.create_dataset('data', data=lr_patches)
    # h5_file.create_dataset('label', data=hr_patches)
    # h5_file.close()
    #这种方式保存太过于耗时
    #删除文件夹
    if os.path.exists(os.path.join(output_path,'lr%s'%up_scale)):
        os.system('rm -rf %s'%(os.path.join(output_path,'lr%s'%up_scale)))
    if os.path.exists(os.path.join(output_path,'hr%s'%up_scale)):
        os.system('rm -rf %s'%(os.path.join(output_path,'hr%s'%up_scale)))
    #创建文件夹
    if not os.path.exists(os.path.join(output_path,'lr%s'%up_scale)):
        os.makedirs(os.path.join(output_path,'lr%s'%up_scale))
    if not os.path.exists(os.path.join(output_path,'hr%s'%up_scale)):
        os.makedirs(os.path.join(output_path,'hr%s'%up_scale))
    for i in tqdm(range(len(lr_patches))):
        #使用image库保存图片
        lr_img = Image.fromarray(lr_patches[i].astype(np.uint8))
        hr_img = Image.fromarray(hr_patches[i].astype(np.uint8))
        if is_residual:
            lr_img.save(os.path.join(output_path,'lr%s_residual'%up_scale,'%s.png'%i))
            hr_img.save(os.path.join(output_path,'hr%s_residual'%up_scale,'%s.png'%i))
        else:
            lr_img.save(os.path.join(output_path,'lr%s'%up_scale,str(i)+'.png'))
            hr_img.save(os.path.join(output_path,'hr%s'%up_scale,str(i)+'.png'))
    print('Done')
if __name__=="__main__":
    dataset_path = "/ssd/home/uguser/pyd/dataset/image_SR/T91"
    output_path = "datasets/T91"
    generate_train_img(dataset_path,output_path)
