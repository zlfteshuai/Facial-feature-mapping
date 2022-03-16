# coding=utf-8
"""
作者：zlf
日期：2021年12月01日
"""
from PIL import Image
import os


def picHandle():
    dirpath = os.listdir('D:/pycharmspace/train')
    i = 37
    for pic in dirpath:
        print(pic)
        im = Image.open('D:/pycharmspace/train/'+pic)
        photo = im.resize((96, 96))
        photo = photo.convert('RGB')
        i+=1
        photo.save(os.path.join('./imgs/1', str(i)+'.jpg'))


picHandle()
