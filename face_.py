# coding=utf-8
"""
作者：zlf
日期：2021年11月18日
"""
import  numpy
from PIL import Image
# 导入处理图像的包：PIL
import face_recognition
# 导入人脸识别的包：face_recognition

# 将jpg文件加载到numpy 数组中
image = face_recognition.load_image_file("2.jpg")
# 查找所有人脸
# 使用HOG模型查找图像中
face_locations = face_recognition.face_locations(image)
# 识别出所有的人脸，返回人脸的列表列表中包括各种信息：上，下，左，右界



# 这个方法已经相当准确了，但还是不如CNN模型那么准确，因为没有使用GPU加速
# 另请参见: find_faces_in_picture_cnn.py
# 使用CNN模型
# face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")




face_landmarks_list = face_recognition.face_landmarks(image)

# 打印：我从图片中找到了 多少 张人脸
print("I found {} face(s) in this photograph.".format(len(face_locations)))

# 循环找到的所有人脸
for face_location in face_locations:
    # 打印每张脸的位置信息
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    # 指定人脸的位置信息，然后显示人脸图片
    face_image = image[top:bottom, left:right]
    # fromarray 将数组转换成图像
    pil_image = Image.fromarray(face_image)
    pil_image.show()


