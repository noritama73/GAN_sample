# https://qiita.com/keiji_dl/items/45a5775a361151f9189d

import torch.nn as nn

"""
datasetsで簡単に手に入るMNIST(0から9の数字60,000枚(28x28ピクセル))を扱うための生成器(Generator)と識別器(Discriminator)の実装をPytorchで行った例を示す。
Pytorchを用いると比較的シンプルに定義することができる。

識別器はnn.Moduleを継承したクラスとして定義する。
入力は28 * 28=784次元に平らにしたイメージの入力を想定し、隠れ層は512次元の全結合層とする。
活性化関数にLeakyReLUを用いて、そのあとはシグモイド関数に入れ二値分類ができるようにしている。

生成器は、ランダムな128次元のノイズを入力し28 x 28ピクセルの画像を生成するように全結合層を３つ利用しており、活性化関数にはReLUを用いている。
"""


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 1)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return nn.Sigmoid()(x)


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 784)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, 28, 28)
        return nn.Tanh()(x)
