import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.cluster import KMeans, AgglomerativeClustering

class LineDetect():
    def __init__(self):
        pass

    def preprocess_images(self, read_cata):
        crop_img_gary = [] # list
        hist = []
        name_set = []

        ### crop images and savefig
        for filename in os.listdir(read_cata): # 类似ls
            name_set.append(filename)

            read_path = os.path.join(read_cata, filename) # 自动加上\符号
            img = cv2.imread(read_path)

            crop_temp = img[45:360, 207:1492, :] # 由loc_axis()得到
            crop_temp = cv2.cvtColor(crop_temp, cv2.COLOR_BGR2RGB)
            crop_temp_gray = cv2.cvtColor(crop_temp, cv2.COLOR_BGR2GRAY)
            crop_img_gary.append(crop_temp_gray)

            ### test imshow cropimg
            # fig = plt.figure()
            # plt.imshow(crop_temp)
            # plt.title(filename)
            # # plt.show()
            # plt.show(block = False) # show.block为False才能close
            # plt.pause(0.5) # 应该是设置pause才能close
            # plt.close('all')

            # crop_temp_gray[crop_temp_gray == 255] = 254 改成254可以克服cv2.calcHist不能检测到255的问题

            hist_temp = cv2.calcHist([crop_temp_gray], [0], None, [257], [0, 256]) #hist是一个np.array 256维度数组
            # 由于cv2.calcHist设置[257]才能得到255的计数，所以要变为256维数的数组
            hist_temp= hist_temp[0:256]
            hist.append(hist_temp)
        self.originHist = hist
        self.originImg = crop_img_gary
        self.name_set = name_set

        return crop_img_gary, hist, name_set

    def visualSave_PltImg(self, img, cata, name_set, Sstring):
        for i, filename in enumerate(name_set):
            filename = Sstring + filename
            save_path = os.path.join(cata, filename)
            cv2.imwrite(save_path, img[i])
            # # plt很难指定savefig尺寸
            # fig = plt.figure()
            # plt.imshow(img[i], cmap = 'gray')
            # plt.axis('off') # 关闭坐标
            # # plt.title(filename) # 不显示title
            # fig.savefig(save_path, bbox_inches='tight', pad_inches=0) # 最大可能的紧缩图片
            # plt.show(block = False)
            # plt.pause(0.5)
            # plt.close('all')
        return
        
    def visualSave_Morph(self, Sstring, cata):
        for i,filename in enumerate(self.name_set):
            filename = Sstring + filename
            save_path = os.path.join(cata, filename)

            fig, axes = plt.subplots(nrows = 4, ncols = 1, figsize=(15, 15))
            
            fig.subplots_adjust(hspace = 1) # hspace 高度间隔
            axes[0].imshow(self.originImg[i], cmap = 'gray')           
            axes[1].imshow(self.thresholdImg[i], cmap = 'gray')
            axes[2].imshow(self.morphOpen[i], cmap = 'gray')            
            axes[3].imshow(self.morphClose[i], cmap = 'gray')
            
            axes[0].set_title("origin gray image")
            axes[1].set_title("thresholdImg")
            axes[2].set_title("morphOpen")
            axes[3].set_title("morphClose")         
            #plt.axes(index = False)
            fig.savefig(save_path)
            plt.show(block = False)
            plt.pause(0.5)
            plt.close('all')

        return

    def visualSave_MorphV2(self, Sstring, cata):
        for i, filename in enumerate(name_set):
            filename = Sstring + filename
            save_path = os.path.join(cata, filename)

            fig, axes = plt.subplots(nrows = 5, ncols = 2, figsize= (30,15))

            fig.subplots_adjust(hspace = 1)
            axes[0,0].imshow(self.originImg[i], cmap = 'gray')
            axes[0,1].imshow(self.thresholdImg[i], cmap = 'gray')
            axes[1,0].imshow(self.dilate[i], cmap = 'gray')
            axes[1,1].imshow(self.erode[i], cmap = 'gray')
            axes[2,0].imshow(self.morphComV4[i], cmap = 'gray')
            axes[2,1].imshow(self.morphComV2[i], cmap = 'gray')
            axes[3,0].imshow(self.morphComV3[i], cmap = 'gray')
            axes[3,1].imshow(self.morphComV1[i], cmap = 'gray')            
            axes[4,0].imshow(self.morphClose[i], cmap = 'gray')
            axes[4,1].imshow(self.morphOpen[i], cmap = 'gray')     

            axes[0,0].set_title("originImg")
            axes[0,1].set_title("thresholdImg")
            axes[1,0].set_title("dilate 3")
            axes[1,1].set_title('erode 3')
            axes[2,0].set_title("morphComV4 clsoe+dlate 3")
            axes[2,1].set_title("morphComV2 open+erode 3")
            axes[3,0].set_title("morphComV3 3")
            axes[3,1].set_title('morphComV1 3')            
            axes[4,0].set_title("morphClose 3")
            axes[4,1].set_title('morphOpen 3')                   
            fig.savefig(save_path)
            # plt.show(block = False)
            # plt.pause(0.5)
            plt.close('all')

        return    

    def post_processing(self):
        self.morphOpen = []
        self.morphClose = []
        self.dilate = []
        self.erode = []
        self.morphComV1 = [] # 1次erode（消除散点），而后1次dilate||开运算
        self.morphComV2 = [] # 1次erode（消除散点），而后闭运算（1次dilate+1次erode）||开运算+erode
        self.morphComV3 = [] # 1次dilate，而后1次erode||闭运算
        self.morphComV4 = [] # 1次dilate，而后开运算（1次erode+1次dilate）||闭运算+dilate

        iter_num = 1
        f_size = 5
        f_sizeV2 = 3
        structure_ele_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (f_size, f_size))
        structure_ele_rectV2 = cv2.getStructuringElement(cv2.MORPH_RECT, (f_sizeV2, f_sizeV2))
        structure_ele_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (f_size, f_size))
        structure_ele_crossV2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (f_sizeV2, f_sizeV2))

        for i,filename in enumerate(name_set):       
            morphRectOpenTemp = cv2.morphologyEx(self.thresholdImg[i], cv2.MORPH_OPEN, structure_ele_rect, iterations=iter_num)
            morphCrossCloseTemp = cv2.morphologyEx(self.thresholdImg[i], cv2.MORPH_CLOSE, structure_ele_cross, iterations=iter_num)
            
            dilateTemp = cv2.dilate(self.thresholdImg[i], structure_ele_rectV2, 1) # 5
            erodeTemp = cv2.erode(self.thresholdImg[i], structure_ele_rectV2, 1) # 3
            morphComTempV1 = cv2.dilate(erodeTemp, structure_ele_rectV2, 1) # 3+5
            morphComTempV2 = cv2.erode(morphComTempV1, structure_ele_rectV2, 1) # 3+5+3
            morphComTempV3 = cv2.erode(dilateTemp, structure_ele_rectV2, 1) # 5+3
            morphComTempV4 = cv2.dilate(dilateTemp, structure_ele_rectV2, 1) # 5+3+5

            self.morphComV1.append(morphComTempV1)
            self.morphComV2.append(morphComTempV2)
            self.morphComV3.append(morphComTempV3)
            self.morphComV4.append(morphComTempV4)

            self.dilate.append(dilateTemp)
            self.erode.append(erodeTemp)
            self.morphOpen.append(morphRectOpenTemp)
            self.morphClose.append(morphCrossCloseTemp)

        return

    def visualSave_Single_Img_Hist(self, Img, Hist, cata, name_set, Sstring):
        for i, filename in enumerate(name_set):
            filename = Sstring + filename
            save_path = os.path.join(cata, filename)

            fig, axes = plt.subplots(nrows = 2, ncols = 1)
            axes[0].imshow(Img[i], cmap = 'gray')
            axes[1].plot(Hist[i])
            axes[0].set_title("gray image")
            axes[1].set_title("hist map")
            fig.savefig(save_path)
            plt.show(block = False)
            plt.pause(0.5)
            plt.close('all')
        return

    def visualSave_Compare_Deno(self, originImg, denoImg, originHist, denoHist, cata, name_set, Sstring):
        for i, filename in enumerate(name_set):
            # assert originImg != denoImg, "same img loc2"
            filename = Sstring + filename
            save_path = os.path.join(cata, filename)

            fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (15,15))
            # fig.tight_layout() # 自动调整子图间隔
            fig.subplots_adjust(hspace = 1) # hspace 高度间隔

            axes[0,0].imshow(originImg[i], cmap = 'gray')
            axes[0,1].plot(originHist[i])
            axes[1,0].imshow(denoImg[i], cmap = 'gray')
            axes[1,1].plot(denoHist[i])

            
            axes[0,0].set_title("origin gray image")
            axes[0,1].set_title("origin hist")
            axes[1,0].set_title("deno image")
            axes[1,1].set_title("deno hist")

            fig.savefig(save_path)
            plt.show(block = False)
            plt.pause(0.5)
            plt.close('all')
        return

    def visualSave_Cluster(self, Sstring, cata):
        for i, filename in enumerate(self.name_set):
            filename = Sstring + filename
            save_path = os.path.join(cata, filename)

            fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (15,15))
            axes[0].imshow(self.originImg[i], cmap = 'gray')
            axes[1].imshow(self.clusterImg[i], cmap = 'gray')
            axes[0].set_title("origin image")
            axes[1].set_title("cluster map")
            fig.savefig(save_path)
            plt.show(block = False)
            plt.pause(0.5)
            plt.close('all')
        return
    
    def visualSave_Threshold(self, Sstring, cata):
        for i, filename in enumerate(self.name_set):
            filename = Sstring + filename
            save_path = os.path.join(cata, filename)

            fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (15,15))
            axes[0].imshow(self.originImg[i], cmap = 'gray')
            axes[1].imshow(self.thresholdImg[i], cmap = 'gray')
            axes[0].set_title("origin image")
            axes[1].set_title("threshold map")
            fig.savefig(save_path)
            plt.show(block = False)
            plt.pause(0.5)
            plt.close('all')
        return

    def denoise_byHist(self, crop_img_gray, name_set):
        deno_img = []
        deno_hist = []

        for i, filename in enumerate(name_set):
            denoTemp = crop_img_gray[i].copy() # copy是新开辟内存空间复制的
            denoTemp[denoTemp < 100] = 0
            histTemp = cv2.calcHist([denoTemp], [0], None, [256], [0, 255])

            deno_img.append(denoTemp)
            deno_hist.append(histTemp)
        self.deno_img = deno_img
        self.deno_hist = deno_hist
        return deno_img, deno_hist
    
    def makedir_(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return

    def denoise_threshold(self):
        '''
        将grayscale取150-160看看
        '''
        self.thresholdImg = []
        for i, filename in enumerate(self.name_set):
            gray = self.originImg[i].copy()
            originHist = self.originHist[i].copy()

            quantity, grayscale = np.unique(originHist, return_index = True)
            print("quantity is: {0}".format(quantity))
            print("grayscale is: {0}".format(grayscale))

            select_val = grayscale[np.where(grayscale<=160)] # 包括150，160
            select_val = select_val[np.where(select_val>=155)] # np.where()没有排序
            select_val = np.sort(select_val)

            grayV1 = np.zeros_like(gray)
            grayV1[grayV1==0] = 255
            for i, value in enumerate(select_val):
                grayV1[gray==value] = i + 1
            
            grayV2 = np.ceil(255/grayV1).astype(np.uint8)
            self.thresholdImg.append(grayV2)
        return        

    def denoise_cluster(self):
        clusterImg = []

        ### calculate cluster_num
        for i, filename in enumerate(self.name_set):
            gray = self.originImg[i].copy()
            originHist = self.originHist[i].copy()

            quantity, grayscale = np.unique(originHist, return_index = True) # 得到index的就是图中所有灰度值的可能取值
            print(quantity) # quantity 增序排列

            ## 根据不同grayscale的个数计算cluster_num
            # originHist[originHist < 40000] = 0
            # unique_v2 = np.unique(originHist)
            # cluster_num = len(unique_v2)
            cluster_num = 8
            print("cluster num: %d" % cluster_num)
            
            ### AgglomerativeCluster
            '''
            层次聚类：
                step1 欧式距离判断最小值
                step2 合并最小值的2类
                step3 计算新的欧氏距离，重复步骤
            '''
            clustering = AgglomerativeClustering(n_clusters = cluster_num)
            res = clustering.fit_predict(grayscale.reshape(-1,1)) # fit_predict返回标签，根据grayscale聚类

            # for i, value in enumerate(grayscale):
            #     gray[gray == value] = (res[i] + 1) * 50 # 重新给像素点赋值，1个类1个值；注意res对应的quantity是增序的            
            # #　res[-1]　res[-2]因为quantity增序，所以是数量最多的，应该是背景，直接给变白
            # gray = np.piecewise(gray,
            #                     [gray == ((res[-1] + 1) * 50), gray == ((res[-2] + 1) * 50), gray == ((res[0] + 1)) * 50],
            #                     [255, 255, 255])

            # gray[gray == 255] = np.amax(grayscale) # 存在255的点，但不知道Hist为什么没识别，反正是噪点直接退化成现有最大的吧
            for i, value in enumerate(grayscale):
                gray[gray == value] = res[i] + 1
            # gray = np.piecewise(gray, [gray == (res[-1] + 1), gray == (res[-2] + 1), gray == (res[0] + 1)], [1,1,1])
            gray = np.piecewise(gray,
                                [gray == (res[-1]+1),
                                gray == (res[0]+1),
                                gray == (res[1]+1),
                                gray == (res[2]+1),
                                gray == (res[3]+1),
                                gray == (res[4]+1),
                                gray == (res[5]+1)],
                                [1, 1, 1, 1, 1, 1, 1])
            grayV2 = np.ceil(255 / gray).astype(np.uint8)
            # cv2.imshow("cluster", grayV2)
            clusterImg.append(grayV2)

        self.clusterImg = clusterImg
        return


if  __name__ == '__main__':
    read_path = r'data\IonTrack\Ion_track_frame-frame'
    save_path_crop_gary = r'data\IonTrack\crop_gary'
    save_path_hist = r'data\IonTrack\hist'
    save_path_denoHist = r'data\IonTrack\deno_hist'
    save_path_cluster = r'data\IonTrack\cluster'
    save_path_threshold = r'data\IonTrack\threshold' # r就不会被转义字符影响
    save_path_morph = r'data\IonTrack\morph'
    save_path_morphV2 = r'data\IonTrack\morph_dilate_erode'

    LineDe = LineDetect()
    LineDe.makedir_(save_path_crop_gary)
    LineDe.makedir_(save_path_hist)
    LineDe.makedir_(save_path_denoHist)
    LineDe.makedir_(save_path_cluster)
    LineDe.makedir_(save_path_threshold)
    LineDe.makedir_(save_path_morph)
    LineDe.makedir_(save_path_morphV2)

    ### crop and gray the originImg
    crop_img_gary, originHist, name_set= LineDe.preprocess_images(read_path)

    assert len(crop_img_gary) == 41,"wrong loc 1" 

    LineDe.visualSave_PltImg(crop_img_gary, save_path_crop_gary, name_set, "_crop_gray_")
    # LineDe.visualSave_Single_Img_Hist(crop_img_gary, originHist, save_path_hist, name_set, "_oriHist_")
    
    ### denoiseImg
    deno_img, denoHist = LineDe.denoise_byHist(crop_img_gary, name_set)
    # LineDe.visualSave_Compare_Deno(crop_img_gary, deno_img, originHist, denoHist, save_path_denoHist, name_set, "_deno_byHist")

    ### threshold
    LineDe.denoise_threshold()
    # LineDe.visualSave_Threshold("_threshold_", save_path_threshold)

    ### cluster
    # LineDe.denoise_cluster()
    # LineDe.visualSave_Cluster("_cluster_", save_path_cluster)

    ### morph
    LineDe.post_processing()
    # LineDe.visualSave_Morph("_morph_", save_path_morph)

    ### dilate|erode only
    LineDe.visualSave_MorphV2("_dilate_erode_", save_path_morphV2)

