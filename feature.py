import dlib
import cv2

# 人脸检测
detector = dlib.get_frontal_face_detector()

# 人脸关键点标注。
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
img = cv2.imread('nrGA6ZxR0E_small.jpg')
# 转灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dets = detector(gray, 1)  # 第二个参数越大，代表讲原图放大多少倍在进行检测，提高小人脸的检测效果。

for d in dets:
    # 使用predictor进行人脸关键点检测 shape为返回的结果
    shape = predictor(gray, d)
    for index, pt in enumerate(shape.parts()):
        # 打印特征点
        # print('Part {}: {}'.format(index, pt))
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 1, (255, 0, 0), 2)
        # 利用cv2.putText标注序号
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(index + 1), pt_pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
