from PIL import Image
import numpy as np

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
    # 读取图片
    img = Image.open(dataset_path)
    # 转换为numpy数组
    img = np.array(img)
    # 生成高斯噪声
    noise = np.random.normal(0, 25, img.shape)
    # 添加高斯噪声
    img_noise = img + noise
    # 保存图片
    img_noise = Image.fromarray(img_noise.astype('uint8')).convert('RGB')
    img_noise.save(output_path)




if __name__=="__main__":
    dataset_path = "/ssd/home/uguser/pyd/dataset/image_SR/T91"
    output_path = "datasets"
    generate_train_img(dataset_path,output_path)