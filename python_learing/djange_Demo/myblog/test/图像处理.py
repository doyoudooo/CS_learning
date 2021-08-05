# 图像预处理
# 1.灰度化
import cv2
# 方法一：读取原图片，再通过函数进行灰度化
img=cv2.imread('21.jpg',1)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('img',img)
# cv2.imshow('gray2',gray)

# 方法二 直接在读取图片的时候灰度化
# img1=cv2.imread('a1.png',0)
# # cv2.imshow('gray1',img1)

# 显示图片


"""二值化处理"""
# 先读入原图像 ，进行灰度化
# 二值化函数
# 其中，第二个参数是判定像素点的临界值。超过了这个点，将会被划分为255，低于这个点，将会被划分为0。具体的参数0~255可以自行根据需要调节。

cv2.threshold(gray,199,255,0,gray)
# # 展示
cv2.namedWindow('Binarization', 0)
cv2.resizeWindow('Binarization', 600, 500)   # 自己设定窗口图片的大小

cv2.imshow('Binarization',gray)
cv2.waitKey(0)


