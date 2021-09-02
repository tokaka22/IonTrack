import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering
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
        dst=self._data[42:378, 167:1494, :]
        gray=cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        hist=cv2.calcHist([gray], [0], None, [256], [0, 255])
        gray[gray==0]=np.argmax(hist)
        return dst,hist,gray

    def method_cluster(self):
        dst, hist, gray = self.pre_processing()
        gray1=gray.copy()
        component, index = np.unique(hist, return_index=True) # 得到index的就是图中所有灰度值的可能取值
        # component, index = np.delete(component, 0), np.delete(index, 0)
        print(component)
        # r, c = gray.shape
        # data = gray.reshape(r * c, -1)  # 注意维数
        hist[hist < 1000] = 0
        unique = np.unique(hist)
        cluster_num = len(unique)
        print('cluster num', cluster_num)

        # 层次聚类
        clustering = AgglomerativeClustering(n_clusters=cluster_num)
        res = clustering.fit_predict(index.reshape(-1, 1))  # 加速聚类

        for i, value in enumerate(index):
            gray[gray == value] = res[i] + 1
        gray = np.piecewise(gray, [gray == (res[-1] + 1), gray == (res[-2] + 1), gray == (res[0] + 1)], [1, 1, 1])
        label = np.ceil(255 / gray).astype(np.uint8)

        # for i, value in enumerate(index):
        #     gray[gray == value] = (res[i] + 1) * 50 # 重新给像素点赋值
        
        # cv2.imshow("gray", gray)
        # gray = np.piecewise(gray,
        #                     [gray == ((res[-1] + 1) * 50), gray == ((res[-2] + 1) * 50), gray == ((res[0] + 1)) * 50],
        #                     [255, 255, 255])
        
        # label = gray.astype(np.uint8)

        cv2.imshow("gray", gray)
        cv2.imshow("label", label)

        # KMeans聚类
        km = KMeans(n_clusters=cluster_num, init='k-means++')
        res1 = km.fit_predict(index.reshape(-1, 1))
        for i, value in enumerate(index):
            gray1[gray1 == value] = (res1[i] + 1) * 50
        gray1 = np.piecewise(gray1, [gray1 == ((res1[-1] + 1) * 50), gray1 == ((res1[-2] + 1) * 50),
                                     gray1 == ((res1[0] + 1)) * 50],
                             [255, 255, 255])
        label1 = gray1.astype(np.uint8)
        # for i, value in enumerate(index):
        #     gray1[gray1 == value] = res1[i] + 1
        # gray1 = np.piecewise(gray1, [gray1 == (res1[-1] + 1), gray1 == (res1[-2] + 1), gray1 == (res1[0] + 1)],
        #                      [1, 1, 1])
        # label1 = np.ceil(255 / gray1).astype(np.uint8)
        return label,label1

    def post_processing(self):
        label,label1=self.method_cluster()
        iter_num = 1
        structure_ele = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        morph = cv2.morphologyEx(label, cv2.MORPH_CLOSE, structure_ele, iterations=iter_num)
        morph1 = cv2.morphologyEx(label1, cv2.MORPH_CLOSE, structure_ele, iterations=iter_num)
        return morph,morph1

    def visualization(self,string):
        dst,hist,gray=self.pre_processing()
        cv2.imshow('pre processing'+string,dst)
        cv2.imwrite('res/pre processing'+string+'.png',dst)
        # plt.title('color hist')
        # self.color_hist(dst)
        # plt.savefig('res/color hist'+string+'.jpg')
        # plt.show()

        plt.plot(hist)
        plt.savefig('res/gray hist'+string+'.jpg')
        plt.show()

        cv2.imshow('pre processing gray'+string, gray)
        cv2.imwrite('res/pre processing gray'+string+'.png', gray)

        label, label1 = self.method_cluster()
        cv2.imshow('cluster agglomerative' + string, label)
        cv2.imwrite('res/cluster agglomerative' + string + '.png', label)
        cv2.imshow('cluster kmeans' + string, label1)
        cv2.imwrite('res/cluster kmeans' + string + '.png', label1)

        morph,morph1=self.post_processing()
        cv2.imshow('morphology agglomerative' + string, morph)
        cv2.imwrite('res/morphology agglomerative' + string+'.png', morph)
        cv2.imshow('morphology kmeans' + string, morph1)
        cv2.imwrite('res/morphology kmeans' + string+'.png', morph1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    src=cv2.imread('data\AdcData-11-6-2-9-26-2D_407.png',1)
    ObjectDetect(src).visualization(str(408))