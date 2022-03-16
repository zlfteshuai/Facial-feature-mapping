# coding=utf-8
"""
作者：zlf
日期：2021年12月01日
"""
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
# 用于网络可视化
from torchvision.models import vgg16
from torchsummary import summary
from torchviz import make_dot
import 人脸特征编码处理


# 定义鉴别器
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        # 使用深度卷积网络作为鉴别器         # 该网络前四层处理的输入是3维数据->32->64->128->256
        self.layer1 = nn.Sequential(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf), nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 8),  nn.LeakyReLU(0.2, inplace=True))
        # 最后全连接层是将 256 * 6 * 6 的维度通过线性变换变成一维
        self.fc = nn.Sequential(nn.Linear(256 * 6 * 6, 1), nn.Sigmoid())

    def forward(self, x):
        out = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        out = self.fc(out.view(-1, 256 * 6 * 6))    # view的功能是将数据转换成 256 * 6 * 6
        out = out.squeeze(-1)
        return out


# 定义生成器
class Generator(nn.Module):
    def __init__(self, nc, ngf, nz, feature_size):
        super(Generator, self).__init__()
        self.prj = nn.Linear(feature_size, nz * 6 * 6)      # 所谓feature_size应该就是输入的噪声的维度
        # nn.Sequential：一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 4), nn.ReLU())
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf), nn.ReLU())
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
                                    nn.Tanh())

    def forward(self, x):
        out = self.prj(x).view(-1, 1024, 6, 6)  # 将输入的噪声预处理成 1024*6*6*others 的形式
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return out


# 图片显示
def img_show(inputs, picname, i):
    # plt.ion()
    inputs = inputs / 2 + 0.5
    inputs = inputs.numpy().transpose((1, 2, 0))
    plt.imshow(inputs)      # 生成彩色图
    # plt.pause(0.1)
    road = 'D:/pycharmspace/mywork/python_finalwork/machine pic/'+str(i)+'/'
    plt.savefig(road + picname + ".jpg")
    plt.close()


# 训练过程
def train(i,d, g, criterion, d_optimizer, g_optimizer, epochs=1, show_every=100, print_every=50,):
    iter_count = 0  # 迭代训练的次数
    for epoch in range(epochs):
        for inputs, _ in train_loader:      # 下划线啥意思没懂
            # 制作真输入和假输入
            real_inputs = inputs  # 真实样本   真实的输入（训练集的图片编码转换成的数组）
            fake_inputs = g(torch.randn(5, 100))  # 把伪造的数据喂给生成器    （通过torch.randn（）模拟的标准正太分布的数据）
        # 制作真标签和假标签
            real_labels = torch.ones(real_inputs.size(0))  # 把预测结果过真实标签
            fake_labels = torch.zeros(5)  # 错误标签
        # 分别用真数据和假数据训练一下鉴别器
            real_outputs = d(real_inputs)       # 把真实的输入喂给鉴别器，让他判断真假
            d_loss_real = criterion(real_outputs, real_labels)      # d_loss_real 是损失函数当输入的是正确标签计算出来的代价

            fake_outputs = d(fake_inputs)       # 把假的输入喂给鉴别器，让他判断真假
            d_loss_fake = criterion(fake_outputs, fake_labels)      # 同理：d_loss_fake 当标签为错误时的损失函数

            d_loss = d_loss_real + d_loss_fake      # 总代价为：两个代价加和
        # 更新鉴别器的权重参数
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            fake_inputs = g(torch.randn(5, 100))        # 另生成器生成图片
            outputs = d(fake_inputs)                    # 把生成器生成的图片传给鉴别器检验真伪
            real_labels = torch.ones(outputs.size(0))   # 给输出的值贴上0的标签
            g_loss = criterion(outputs, real_labels)    # 根据生成的图片的是否能欺骗鉴别器来优化生成器
        # 更新生成器的权重参数
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if iter_count % show_every == 0:        # 训练的次数为展示的整数次时，显示并保存一下生成器生成的假图片。并告知损失函数的数值
                print('Epoch:{}, Iter:{}, D:{:.4}, G:{:.4}'.format(epoch,
                                                                 iter_count,
                                                                 d_loss.item(),
                                                                 g_loss.item()))
                picname = "Epoch_" + str(epoch) + "Iter_" + str(iter_count)
                img_show(torchvision.utils.make_grid(fake_inputs.data), picname, i)

            if iter_count % print_every == 0:
                print('Epoch:{}, Iter:{}, D:{:.4}, G:{:.4}'.format(epoch,
                                                                 iter_count,
                                                                 d_loss.item(),
                                                                 g_loss.item()))
            iter_count += 1

            print(iter_count, 'Finished Training！')
    save_params(d, g, i)


def save_params(d, g, i):
    # 存储训练好的网络，i代表了存储网络的路径，epochs代表了网络的epochs
    torch.save(d.state_dict(), 'D:/pycharmspace/mywork/模型训练参数/d/d_' + str(i))
    torch.save(g.state_dict(), 'D:/pycharmspace/mywork/模型训练参数/g/g_' + str(i))


def show_layer():
    d = vgg16()  # 实例化网络，可以换成自己的网络
    summary(d, (3, 64, 64))  # 输出网络结构
    g = vgg16()
    summary(g, (3, 64, 64))


def load_param(i):
    d.load_state_dict(torch.load('D:/pycharmspace/mywork/模型训练参数/d/d_' + str(i)))
    g.load_state_dict(torch.load('D:/pycharmspace/mywork/模型训练参数/g/g_' + str(i)))
    see_pic()


def see_pic():
    fake_pic = g(torch.randn(5, 100))  # 另生成器生成图片
    fake_pic = torchvision.utils.make_grid(fake_pic)
    plt.ion()
    fake_pic = fake_pic / 2 + 0.5
    fake_pic = fake_pic.numpy().transpose((1, 2, 0))
    plt.imshow(fake_pic)
    plt.pause(3)
    plt.close()


def face():
    fake_pic = g(人脸特征编码处理.faceCode)  # 另生成器生成图片
    fake_pic = torchvision.utils.make_grid(fake_pic)
    plt.ion()
    fake_pic = fake_pic / 2 + 0.5
    fake_pic = fake_pic.numpy().transpose((1, 2, 0))
    plt.imshow(fake_pic)
    plt.pause(30)
    plt.close()


def see_net():
    g = Generator(3, 128, 1024, 100)
    d = Discriminator(3, 32)
    y = d(g(torch.randn(5, 100)))
    # y = g(torch.randn(5, 100))
    vise = make_dot(y, params=dict(d.named_parameters()))
    vise.view()


# 主程序
if __name__ == '__main__':
    # 串联多个变换操作
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 依概率p水平翻转，默认p=0.5
        transforms.ToTensor(),  # 转为tensor，并归一化至[0-1]
        # 标准化，把[0-1]变换到[-1,1]，其中mean和std分别通过(0.5,0.5,0.5)和(0.5,0.5,0.5)进行指定。
        # 原来的[0-1]最小值0变成(0-0.5)/0.5=-1，最大值1变成(1-0.5)/0.5=1
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 参数data_transform：对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片。
    train_set = datasets.ImageFolder('imgs', data_transform)  # 把imgs中的文件读入到train_set中，由字典形成的列表组成
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=True, num_workers=0)  # 数据加载

    inputs, _ = next(iter(train_loader))        # 啥意思？？？？？？？？？
    # make_grid的作用是将若干幅图像拼成一幅图像
    # img_show(torchvision.utils.make_grid(inputs), "RealDataSample")

    # 初始化鉴别器和生成器
    d = Discriminator(3, 32)    # 两个参数： 色彩通道数3：RGB三色通道  特征图的深度：32 （通过卷积层生成了32个卷积核，每个卷积核生成一个特征图 即：过滤器：filter）
    g = Generator(3, 128, 1024, 100)  # 四个参数： 3：RGB三色通道  128：生成器的过滤器  1024：生成器的输入  100：生成的噪声的维度（特征的大小）

    criterion = nn.BCELoss()  # 损失函数
    lr = 0.0003  # 学习率
    d_optimizer = torch.optim.Adam(d.parameters(), lr=lr)  # 定义鉴别器的优化器
    g_optimizer = torch.optim.Adam(g.parameters(), lr=lr)  # 定义生成器的优化器
    #
    #
    #
    #
    # train(8,d, g, criterion, d_optimizer, g_optimizer, epochs=600) # 其中6表示生成图片的保存路径，保存训练参数的路径
    # 六个参数：1：鉴别器 2：生成器 3：损失函数 4：鉴别器的优化算法 5：生成器的优化算法 6：遍历训练集的次数：300次
    load_param(7)
    face()
    # see_net()





