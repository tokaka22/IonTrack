import cv2
import os
'''
获取结构元
cv2.getStructuringElement(shape=,ksize=,anchor=)
shape:cv2.MORPH_RECT为矩形结构元，cv2.MORPH_ELLIPSE为椭圆结构元，
cv2.MORPH_CROSS为十字交叉结构元
ksize:结构元的尺寸
anchor：结构元的锚点
腐蚀：取结构元中像素值最小
cv2.erode(src=,kernel=,anchor=,iterations=,borderType=,borderValue=)
kernel：结构元
anchor：结构元的锚点
iterations：腐蚀操作的次数
borderType：边界扩充类型
borderValue：边界扩充值
膨胀
cv2.dilate(src, kernel, dst=None, anchor=None, iterations=None, borderType=None, borderValue=None)
形态学操作的函数
cv2.morphologyEx(src, op, kernel, dst=None, anchor=None, iterations=None, borderType=None, borderValue=None)
op:形态学处理的各种运算,cv2.MORPH_OPEN为开运算（先腐蚀后膨胀），cv2.MORPH_CLOSE为闭运算
cv2.MORPH_ERODE为腐蚀，cv2.MORPH_DILATE为膨胀，cv2.MORPH_TOPHAT为顶帽操作(校正光照不均匀)，
cv2.MORPH_BLACKHAT为底帽运算，cv2.MORPH_GRADIENT为形态梯度
kernel：结构元
anchor：结构元的锚点
iterations：操作的次数
borderType：边界扩充类型
borderValue：边界扩充值
'''
save_cata = r"data\IonTrack\morphHyper"
fileName = 'init'

def initArgs():
    args = []

    src=cv2.imread(r'data\IonTrack\crop_gary\_crop_gray_AdcData-11-6-2-32-15-2D_24.png',0)
    min_radius,min_it=1,1#结构元半径和迭代次数
    print("cv2.MORPH_XXX value：{},{},{}".format(cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE))#形式参数序号，可以用0，1，2去表示
    min_kernel,max_kernel=0,2
    max_radius,max_it=20,20
    morphology=[cv2.MORPH_ERODE,cv2.MORPH_DILATE,cv2.MORPH_OPEN,cv2.MORPH_CLOSE,
                cv2.MORPH_GRADIENT,cv2.MORPH_TOPHAT,cv2.MORPH_BLACKHAT]#这几个全局常量就是1，2，3之类的

    args.extend((src, min_radius, max_radius, min_it, max_it, min_kernel, max_kernel, morphology)) # only need 1 parameter
    return args

class morphCustom():
    def __init__(self, args):
        # args：'src, min_radius, max_radius, min_it, max_it, min_kernel, max_kernel, morphology'
        self.args = args
        self.src = args[0]
        self.min_radius = args[1]
        self.max_radius = args[2]
        self.min_it = args[3]
        self.max_it = args[4]
        self.min_kernel = args[5]
        self.max_kernel = args[6]
        self.morphology = args[7]
        # self.fileName = 'init'
        self.nameList = ['ERODE','DILATE','OPEN','CLOSE',\
                'GRADIENT','TOPHAT','BLACKHAT']
        return
    
    def nothing(self, *args):#*args表示任何多个无名参数，它本质是一个tuple，tuple即不可改变的List
        pass
    
    def keyboardChange(self):
        try:
            num = int(input("1.cv2.MORPH_ERODE \n2.cv2.MORPH_DILATE \n3.cv2.MORPH_OPEN \n4.cv2.MORPH_CLOSE \
                \n5.cv2.MORPH_GRADIENT \n6.cv2.MORPH_TOPHAT \n7.cv2.MORPH_BLACKHAT \nchoose:"))
        except EOFError as e:
            pass
        # num = 1
        select = num - 1
        return self.morphology[select]

    # def iterMorph(self, morphCustom):
    #     self.morphCustomOther = morphCustom
    #     return

    def changeSrc(self, newSrc):
        self.src = newSrc
        return

    # def newFileName(self, newName):
    #     self.fileName = self.fileName + '+' + newName
    #     return

    def morph(self):
        self.morphType = self.keyboardChange()
        addFileName = self.nameList[self.morphType]
        # print(fileName)
        global fileName # 全局变量
        fileName = fileName + '+' +addFileName
        saveName = fileName + '.png'
        save_path = os.path.join(save_cata, saveName)
        print("fileName: {}".format(fileName))

        # print(self.morphType)
        cv2.namedWindow('morphology '+str(self.morphType), cv2.WINDOW_AUTOSIZE)#WINDOW_NORMAL可以自主调整windows大小
        cv2.createTrackbar('radius','morphology '+str(self.morphType), self.min_radius, self.max_radius, self.nothing)#创建调节结构元半径的进度条
        cv2.createTrackbar('it','morphology '+str(self.morphType), self.min_it, self.max_it, self.nothing)#创建调节迭代次数的进度条
        cv2.createTrackbar('kernel_style', 'morphology ' + str(self.morphType), self.min_kernel, self.max_kernel, self.nothing)  # 创建调节结构元类型的进度条
        while True:
            self.radius=cv2.getTrackbarPos('radius','morphology '+str(self.morphType))#获取进度条上当前radius的值
            self.it=cv2.getTrackbarPos('it','morphology '+str(self.morphType))
            self.kernel_style= cv2.getTrackbarPos('kernel_style', 'morphology ' + str(self.morphType))
            s=cv2.getStructuringElement(self.kernel_style, (2*self.radius+1,2*self.radius+1))
            d=cv2.morphologyEx(self.src,self.morphType,s,iterations=self.it)

            cv2.imshow('morphology '+str(self.morphType),d)
            ch=cv2.waitKey(5)
            # print(ch)
            if ch==32:#space键进行下一个
                d[d<50] = 0
                self.morphCustomOther = morphCustom(self.args)
                self.morphCustomOther.changeSrc(d)
                # self.morphCustomOther.newFileName(addFileName)
                self.morphCustomOther.morph()
            if ch==27:#按下ESC键退出内循环
                break
            if ch==0:#存储当前图像
                cv2.imwrite(save_path, d)
                print("save finished")
        cv2.destroyAllWindows()
        return
    

if __name__ == '__main__':
    args = initArgs()

    morphCustom_0 = morphCustom(args)
    morphCustom_0.morph()
    

'''
for i in morphology:
    cv2.namedWindow('morphology '+str(i), cv2.WINDOW_AUTOSIZE)#WINDOW_NORMAL可以自主调整windows大小
    cv2.createTrackbar('radius','morphology '+str(i),radius,max_radius,nothing)#创建调节结构元半径的进度条
    cv2.createTrackbar('it','morphology '+str(i),it,max_it,nothing)#创建调节迭代次数的进度条
    cv2.createTrackbar('kernel_style', 'morphology ' + str(i), kernel_style, max_kernel, nothing)  # 创建调节结构元类型的进度条
    while True:
        radius=cv2.getTrackbarPos('radius','morphology '+str(i))#获取进度条上当前radius的值
        it=cv2.getTrackbarPos('it','morphology '+str(i))
        kernel_style= cv2.getTrackbarPos('kernel_style', 'morphology ' + str(i))
        s=cv2.getStructuringElement(kernel_style,(2*radius+1,2*radius+1))
        d=cv2.morphologyEx(src,i,s,iterations=it)
        cv2.imshow('morphology '+str(i),d)
        ch=cv2.waitKey(5)

        if ch==27:#按下ESC键退出内循环
            break
cv2.destroyAllWindows()
'''