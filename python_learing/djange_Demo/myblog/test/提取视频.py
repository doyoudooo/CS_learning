import os
import cv2
import windnd
from tkinter import *


def video_to_imgs(sourceFile):
    video_path = os.path.join("", "", sourceFile + '.MP4')
    times = 0
    frameFrequency = 30  # 在此处更改每X帧截取一张
    outPutDirName = '' + sourceFile + '\\'
    if not os.path.exists(outPutDirName):
        os.makedirs(outPutDirName)
    cap = cv2.VideoCapture(video_path)
    while True:
        times += 1
        res, image = cap.read()
        if not res:
            break
        if times % frameFrequency == 0:
            cv2.imencode('.jpg', image)[1].tofile(outPutDirName + str(times) + '.jpg')
            print(outPutDirName + str(times) + '.jpg')
    cap.release()
    print('已输出至' + sourceFile + '\\')


def accept_video(files):
    print(files[0][0:-4].decode('GBK'))
    video_to_imgs(files[0][0:-4].decode('GBK'))


tk = Tk()
tk.wm_attributes('-topmost', 1)
tk.title("视频逐帧提取丨吾爱破解")
windnd.hook_dropfiles(tk, func=accept_video)
tk.mainloop()