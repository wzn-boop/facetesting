# 人脸识别
import os
import dlib  # 人脸处理的库 Dlib
import time
import numpy as np  # 数据处理的库 numpy
from cv2 import cv2 as cv2  # 图像处理的库 OpenCv
import pandas as pd  # 数据处理的库 Pandas
from skimage import io

# 人脸识别模型，提取128D的特征矢量
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

path_images_from_camera = "celeb_data_30/"
people = os.listdir(path_images_from_camera)

# 计算两个128D向量间的欧式距离
def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist


# 处理存放所有人脸特征的 csv
path_features_known_csv = "features.csv"
csv_rd = pd.read_csv(path_features_known_csv, header=None)

# 用来存放所有录入人脸特征的数组
features_known_arr = []

# 读取已知人脸数据
for i in range(csv_rd.shape[0]):
    features_someone_arr = []
    for j in range(0, len(csv_rd.iloc[i, :])):
        features_someone_arr.append(csv_rd.iloc[i, :][j])
    features_known_arr.append(features_someone_arr)
print("Faces in Database：", len(features_known_arr))

# Dlib 检测器和预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

path_test_photos = 'test/'
photos_list = os.listdir(path_test_photos)
if photos_list:
    for i in range(len(photos_list)):
        img_rd = io.imread(path_test_photos + '/' + photos_list[i])
        # 取灰度
        img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

        # 人脸数 faces
        faces = detector(img_gray, 0)

        # 待会要写的字体 font to write later
        font = cv2.FONT_HERSHEY_COMPLEX

        # 存储当前图片中捕获到的所有人脸的坐标/名字
        # the list to save the positions and names of current faces captured
        pos_namelist = []
        name_namelist = []

        # 检测到人脸 when face detected
        if len(faces) != 0:
            # 获取当前图片的所有人脸的特征，存储到 features_cap_arr
            features_cap_arr = []
            for j in range(len(faces)):
                shape = predictor(img_rd, faces[j])
                features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))

            # 遍历捕获到的图像中所有的人脸
            for k in range(len(faces)):
                print("##### person", k + 1, "#####")
                # 先默认所有人不认识，是 unknown
                name_namelist.append("unknown")

                # 每个捕获人脸的名字坐标 the positions of faces captured
                pos_namelist.append(
                    tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                # 对于某张人脸，遍历所有存储的人脸特征
                e_distance_list = []
                for h in range(len(features_known_arr)):
                    # 如果 person_X 数据不为空
                    if str(features_known_arr[h][0]) != '0.0':
                        print("with person", str(h + 1), "the e distance: ", end='')
                        e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[h])
                        print(e_distance_tmp)
                        e_distance_list.append(e_distance_tmp)
                    else:
                        # 空数据 person_X
                        e_distance_list.append(999999999)
                # 找出最接近的一个人脸数据是第几个
                similar_person_num = e_distance_list.index(min(e_distance_list))
                print("Minimum e distance with person", int(similar_person_num) + 1)

                # 计算人脸识别特征与数据集特征的欧氏距离
                # 距离小于0.4则标出为可识别人物
                if min(e_distance_list) < 0.4:
                    print("find it!")
                    name_namelist[k] = people[e_distance_list.index(min(e_distance_list))]
                else:
                    print("Unknown person")
            # 在人脸框下面写人脸名字
            # write names under rectangle
            # for i in range(len(faces)):
            #     cv2.putText(img_rd, name_namelist[i], pos_namelist[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

        print("Faces in photos[", i + 1, "]:", name_namelist, "\n")

        # 窗口显示 show with opencv
        cv2.imshow("img", img_rd)
        cv2.waitKey(0)

# 删除建立的窗口 delete all the windows
cv2.destroyAllWindows()
