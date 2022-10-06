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

def generate_train_img(dataset_paths:list,output_path,up_scale=3,patch_size=11,is_residual=False,phase="train"):
    """
    生成训练集
    :param dataset_path: 训练集路径
    :param output_path: 生成的训练集路径
    :return:
    """
    # 获取数据集路径
    img_list = []
    for dataset_path in dataset_paths:
        img_path_list = sorted(os.listdir(dataset_path))

        #数据增强
        #We augment the data in two ways. 1) Scaling: each image is downscaled with the factor 0.9, 0,8, 0.7 and 0.6. 2) Rotation: each image is rotated with the degree of 90, 180 and 270.
        if phase == "train":
            for img_path in tqdm(img_path_list):
                img = Image.open(os.path.join(dataset_path,img_path))
                for scale in range(6,10):
                    for roatation in [0,90,180,270]:
                        new_img = img.resize((img.size[0]*scale//10,img.size[1]*scale//10),Image.BICUBIC)
                        new_img = img.rotate(roatation)
                        img_list.append(new_img)
        else:
            for img_path in tqdm(img_path_list):
                img = Image.open(os.path.join(dataset_path,img_path))
                img_list.append(img)
    #采样
    #To prepare the training data, we first downsample the original training images by the desired scaling factor n to form the LR images. Then we crop the LR training images into a set of fsub × fsub -pixel sub-images with a stride k. The corresponding HR sub-images (with size (n*fsub)^2) are also cropped from the ground truth images. These LR/HR sub-image pairs are the primary training data.
    data_patches = []
    label_patches = []
    for label in tqdm(img_list):
            # label = label.convert('YCbCr')
            # #只取Y通道的数据
            # label = label.split()[0]
            label_width = (label.width // up_scale) * up_scale
            label_height = (label.height // up_scale) * up_scale
            label = label.resize((label_width, label_height), resample=Image.BICUBIC)
            data = label.resize((label.width // up_scale, label_height // up_scale), resample=Image.BICUBIC)
            unscale = data.resize((data.width * up_scale, data.height * up_scale), resample=Image.BICUBIC)
            label = np.array(label).astype(np.float32)
            data = np.array(data).astype(np.float32)
            unscale = np.array(unscale).astype(np.float32)
            if is_residual:
                label = label - unscale
            for i in range(0, data.shape[0] - patch_size + 1, up_scale):
                for j in range(0, data.shape[1] - patch_size + 1, up_scale):
                    data_patches.append(data[i:i+patch_size, j:j+patch_size])
                    label_patches.append(label[i*up_scale:i*up_scale+patch_size*up_scale, j*up_scale:j*up_scale+patch_size*up_scale])
    #保存数据集
    data_patches = np.array(data_patches)
    label_patches = np.array(label_patches)
    #这种方式保存速度较快，但是框架里没有相应的读取方式
    # if is_residual:
    #     h5_file = h5py.File(os.path.join(output_path,"train_x{}_residual".format(up_scale)), 'w')
    # else:
    #      h5_file = h5py.File(os.path.join(output_path,"train_x{}".format(up_scale)), 'w')
    # h5_file.create_dataset('data', data=lr_patches)
    # h5_file.create_dataset('label', data=hr_patches)
    # h5_file.close()
    #这种方式保存太过于耗时
    data_name = "data_x{}_residual".format(up_scale) if is_residual else "data_x{}".format(up_scale)
    label_name = "label_x{}_residual".format(up_scale) if is_residual else "label_x{}".format(up_scale)
    #删除文件夹
    if os.path.exists(os.path.join(output_path,data_name)):
        os.system('rm -rf %s'%(os.path.join(output_path,data_name)))
    if os.path.exists(os.path.join(output_path,label_name)):
        os.system('rm -rf %s'%(os.path.join(output_path,label_name)))
    #创建文件夹
    if not os.path.exists(os.path.join(output_path,data_name)):
        os.makedirs(os.path.join(output_path,data_name))
    if not os.path.exists(os.path.join(output_path,label_name)):
        os.makedirs(os.path.join(output_path,label_name))
    for i in tqdm(range(len(data_patches))):
        #使用image库保存图片
        lr_img = Image.fromarray(data_patches[i].astype(np.uint8))
        hr_img = Image.fromarray(label_patches[i].astype(np.uint8))
        lr_img.save(os.path.join(output_path,data_name,'%s.png'%i))
        hr_img.save(os.path.join(output_path,label_name,'%s.png'%i))
    print('Done')
if __name__=="__main__":
    dataset_path = "/ssd/home/uguser/pyd/dataset/image_SR/T91"
    output_path = "datasets/T91"
    # dataset_path = "datasets/Set14/original"
    # output_path = "datasets/Set14/modified"
    generate_train_img(dataset_path,output_path,phase="train",is_residual=True)
