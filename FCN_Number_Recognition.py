#!/usr/bin/python
##
# _3用全连接网络FCN实现手写数字识别
# 数据采用业界非常流行的 MNIST 数据集
##

import struct
from _3FCNetwork import *
from datetime import datetime
# 获取手写数据。
# 28*28的图片对象。每个图片对象根据需求是否转化为长度为784的横向量
# 每个对象的标签为0-9的数字，one-hot编码成10维的向量

# 数据加载器基类。派生出图片加载器和标签加载器
class Loader(object):
    def __init__(self, path, count):
        '''
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        '''
        self.path = path
        self.count = count
    def get_file_content(self):
        '''
        读取文件内容
        '''
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

    # 将unsigned byte字符转换为整数。python3中bytes的每个分量读取就会变成int
    #def to_int(self, byte):
    #    '''
    #    将unsigned byte字符转换为整数
    #    '''
    #    return struct.unpack('B', byte)[0]


# 图像数据加载器
class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        内部函数，从文件中获取图像
        '''
        start = index * 28 * 28 + 16 # 文件头16字节，后面每28*28个字节为一个图片数据
        picture = []
        for i in range(28):
            picture.append([])# 图片添加一行像素
            for j in range(28):
                byte1 = content[start + i * 28 + j]
                picture[i].append(byte1)  # python3中本来就是int
                #picture[i].append(
                #    self.to_int(content[start + i * 28 + j]))
        return picture # 图片为[[x,x,x..][x,x,x...][x,x,x...][x,x,x...]]的列表

    # 将图像数据转化为784的行向量形式
    def get_one_sample(self, picture):
        '''
        内部函数，将图像转化为样本的输入向量
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    # 加载数据文件，获得全部样本的输入向量。onerow表示是否将每张图片转化为行向量
    def load(self):
        '''
        加载数据文件，获得全部样本的输入向量
        '''
        content = self.get_file_content()# 获取文件字节数组
        data_set = []
        for index in range(self.count):#遍历每一个样本
            data_set.append(
                self.get_one_sample(# 从样本数据集中获取第index个样本的图片数据，返回的是二维数组
                    self.get_picture(content, index)))# 将图像转化为一维向量形式
        return data_set
# 标签数据加载器
class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        content = self.get_file_content()# 获取文件字节数组
        labels = []
        for index in range(self.count):#遍历每一个样本
            labels.append(self.norm(content[index + 8]))# 文件头有8个字节one-hot编码
        return labels
    def norm(self, label):
        '''
        内部函数，将一个值转换为10维标签向量
        '''
        label_vec = []
        #label_value = self.to_int(label)
        label_value = label  # python3中直接就是int

        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec
def get_training_data_set():
    '''
    获得训练数据集
    '''
    image_loader = ImageLoader(r'MNIST_DATA\train-images.idx3-ubyte', 100)
    label_loader = LabelLoader(r'MNIST_DATA\train-labels.idx1-ubyte', 100)
    return image_loader.load(), label_loader.load()
def get_test_data_set():
    '''
    获得测试数据集
    '''
    image_loader = ImageLoader(r'MNIST_DATA\t10k-images.idx3-ubyte', 10)
    label_loader = LabelLoader(r'MNIST_DATA\t10k-labels.idx1-ubyte', 10)
    return image_loader.load(), label_loader.load()

# 网络的输出是一个10维向量，这个向量第个(从0开始编号)元素的值最大，那么就是网络的识别结果。下面是代码实现：
def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

# 我们使用错误率来对网络进行评估，下面是代码实现：
def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)



# 最后实现我们的训练策略：每训练10轮，评估一次准确率，当准确率开始下降时终止训练。下面是代码实现：
def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    print('样本数据集的个数：%d' % len(train_data_set))
    print('测试数据集的个数：%d' % len(test_data_set))
    network = Network([784, 300, 10])
    while True: # 迭代至准确率开始下降
        epoch += 1 # 记录迭代次数
        network.train(train_labels, train_data_set, 0.3, 1)# 使用训练集进行训练。0.3为学习速率，1为迭代次数
        print('%s epoch %d finished' % (datetime.datetime.now(), epoch))# 打印时间和迭代次数
        if epoch % 10 == 0:# 每训练10次，就计算一次准确率
            error_ratio = evaluate(network, test_data_set, test_labels)# 计算准确率
            print('%s after epoch %d, error ratio is %f' % (datetime.datetime.now(), epoch, error_ratio))# 打印输出错误率
            if error_ratio > last_error_ratio:# 如果错误率开始上升就不再训练了。
                break
            else:
                print('错误率：', last_error_ratio)
                last_error_ratio = error_ratio# 否则继续训练
    index = 0
    for layer in network.layers:
        np.savetxt('MNIST—W' + str(index), layer.W)
        np.savetxt('MNIST—b' + str(index), layer.b)
        index += 1
        print(layer.W)
        print(layer.b)


if __name__ == '__main__':
    train_and_evaluate()