# coding=utf-8
"""
作者：zlf
日期：2021年12月06日
"""
import pandas as pd
import numpy as np
import torch
from scipy import stats, special
import tensorflow
import face_recognition
image = face_recognition.load_image_file("feng.jpg")
face_encoding = face_recognition.face_encodings(image)[0]
#
face_encoding = face_encoding[:100]
faceCode = face_encoding
faceCode = torch.from_numpy((faceCode))
faceCode = torch.cat((faceCode,faceCode,faceCode,faceCode,faceCode),0)
faceCode = torch.reshape(faceCode,(5,100))
faceCode = faceCode.to(torch.float32)
