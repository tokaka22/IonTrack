import cv2
import numpy as np
import matplotlib.pyplot as plt

class ObjectDetect():
    def __init__(self,data):
        self._data=data

    def color_hist(self,img):
        color=('b','g','r')
        for i,color in enumerate(color):
            hist=cv2.calcHist([img],[i],None,[256],[0,255])
            plt.plot(hist,color)
            plt.xlim([0,255])

    def pre_processing(self):
        dst=self._data
        gray=cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        hist=cv2.calcHist([gray], [0], None, [256], [0, 255]) # hist是一个shape为(256,1)的数组，表示0-255每个像素值对应的像素个数，下标即为相应的像素值
        gray[gray==0]=np.argmax(hist) # 消除灰度值为0的地方（将其变为出现频率最高的gray值）
        return dst,hist,gray

    def visualization(self,string):
        dst,hist,gray=self.pre_processing()
        cv2.imshow('pre_processing'+string,dst)
        cv2.imwrite('res/pre_processing'+string+'.png',dst)
        plt.figure(1)
        plt.title('color hist')
        self.color_hist(dst)
        plt.savefig('res/color_hist'+string+'.jpg')
        plt.show()

        plt.figure(2)
        plt.plot(hist)
        plt.savefig('res/gray_hist'+string+'.jpg')
        plt.show()

        cv2.imshow('pre_processing_gray'+string, gray)
        cv2.imwrite('res/pre_processing_gray'+string+'.png', gray)
        cv2.waitKey(0)



if __name__ == "__main__":
    # src=cv2.imread('E:\tokaka\recent_learn\line_detection\code\test.png', 1) # 默认就是1，彩色格式
    src = cv2.imread("data\IonTrack\Ion_track_ROI\AdcData-11-6-2-32-15-2D_24.png")
    ObjectDetect(src).visualization(str(407))