# 解决：如何把一个黑白的照片上色

开始解决吧！

## 项目简介

本项目基于paddlepaddle，结合残差网络（ResNet）,通过监督学习的方式，训练模型将黑白图片转换为彩色图片

### 1.1 残差网络（ResNet）

ResNet(Residual Network) 是2015年ImageNet图像分类、图像物体定位和图像物体检测比赛的冠军。针对随着网络训练加深导致准确度下降的问题，ResNet提出了残差学习方法来减轻训练深层网络的困难。在已有设计思路(BN, 小卷积核，全卷积网络)的基础上，引入了残差模块。每个残差模块包含两条路径，其中一条路径是输入特征的直连通路，另一条路径对该特征做两到三次卷积操作得到该特征的残差，最后再将两条路径上的特征相加。

残差模块如图所示，左边是基本模块连接方式，由两个输出通道数相同的3x3卷积组成。右边是瓶颈模块(Bottleneck)连接方式，之所以称为瓶颈，是因为上面的1x1卷积用来降维(图示例即256->64)，下面的1x1卷积用来升维(图示例即64->256)，这样中间3x3卷积的输入和输出通道数都较小(图示例即64->64)。

![image-01](https://i.loli.net/2020/08/28/UzFmP7YidZ1G2wt.png)

### 1.2 项目设计思路及主要解决问题

> 设计思路：通过训练网络对大量样本的学习得到经验分布（例如天空永远是蓝色的，草永远是绿色的），通过经验分布推得黑白图像上各部分合理的颜色

> 主要解决问题：大量物体颜色并不是固定的也就是物体颜色具有多模态性（例如：苹果可以是红色也可以是绿色和黄色）。通常使用均方差作为损失函数会让具有颜色多模态属性的物体趋于寻找一个“平均”的颜色（通常为淡黄色）导致着色后的图片饱和度不高。

### 1.3 本文主要特征

   * 将Adam优化器beta1参数设置为0.8，具体请参考[原论文](https://arxiv.org/abs/1412.6980)
   * 将BatchNorm批归一化中momentum参数设置为0.5
   * 采用基本模块连接方式
   * 为抑制多模态问题，在均方差的基础上重新设计损失函数

### 1.4 数据集介绍（ImageNet）

ImageNet项目是一个用于视觉对象识别软件研究的大型可视化数据库。超过1400万的图像URL被ImageNet手动注释，以指示图片中的对象;在至少一百万个图像中，还提供了边界框。ImageNet包含2万多个类别; [2]一个典型的类别，如“气球”或“草莓”，包含数百个图像。第三方图像URL的注释数据库可以直接从ImageNet免费获得;但是，实际的图像不属于ImageNet。自2010年以来，ImageNet项目每年举办一次软件比赛，即ImageNet大规模视觉识别挑战赛（ILSVRC），软件程序竞相正确分类检测物体和场景。 ImageNet挑战使用了一个“修剪”的1000个非重叠类的列表。2012年在解决ImageNet挑战方面取得了巨大的突破，被广泛认为是2010年的深度学习革命的开始。（来源：百度百科）

## 开始实战

## 使用Shell命令对数据集进行初步处理

```shell
mv data/data9402/train.* data/data9244/
mkdir data/tar
mkdir work/train
mkdir work/test
tar xf data/data9244/ILSVRC2012_img_val.tar -C work/test/
cd data/tar/;cat ../data9244/train.tar* | tar -x
cd ./work/train/;ls ../data/tar/*.tar | xargs -n1 tar xf
#显示work/train中图片数量（可选）
find work/train -type f | wc -l
```

## 预处理

### 3.1预处理-采用多线程对训练集中单通道图删除

```python
import os
import imghdr
import numpy as np
from PIL import Image
import threading

'''多线程将数据集中单通道图删除'''
def cutArray(l, num):
  avg = len(l) / float(num)
  o = []
  last = 0.0

  while last < len(l):
    o.append(l[int(last):int(last + avg)])
    last += avg

  return o
  
def deleteErrorImage(path,image_dir):
    count = 0
    for file in image_dir:
        try:
            image = os.path.join(path,file)
            image_type = imghdr.what(image)
            if image_type is not 'jpeg':
                os.remove(image)
                count = count + 1
                #print('已删除：' + image)

            img = np.array(Image.open(image))
            if len(img.shape) is 2:
                os.remove(image)
                count = count + 1 
                #print('已删除：' + image)
        except Exception as e:
            print(e)
    print('done!')
    print('已删除数量：' +  str(count))

class thread(threading.Thread):
    def __init__(self, threadID, path, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.path = path
        self.files = files
    def run(self):
        deleteErrorImage(self.path,self.files)
            
if __name__ == '__main__':
    path = './work/train/'
    files =  os.listdir(path)
    files = cutArray(files,8)
    t1 = threading.Thread(target=deleteErrorImage,args=(path,files[0]))
    t2 = threading.Thread(target=deleteErrorImage,args=(path,files[1]))
    t3 = threading.Thread(target=deleteErrorImage,args=(path,files[2]))
    t4 = threading.Thread(target=deleteErrorImage,args=(path,files[3]))
    t5 = threading.Thread(target=deleteErrorImage,args=(path,files[4]))
    t6 = threading.Thread(target=deleteErrorImage,args=(path,files[5]))
    t7 = threading.Thread(target=deleteErrorImage,args=(path,files[6]))
    t8 = threading.Thread(target=deleteErrorImage,args=(path,files[7]))
    threadList = []
    threadList.append(t1)
    threadList.append(t2)
    threadList.append(t3)
    threadList.append(t4)
    threadList.append(t5)
    threadList.append(t6)
    threadList.append(t7)
    threadList.append(t8)
    for t in threadList:
        t.setDaemon(True)
        t.start()
        t.join()
```

### 3.2预处理-采用多线程对图片进行缩放后裁切到512*512分辨率

```python
from PIL import Image
import os.path
import os
import threading
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''多线程将图片缩放后再裁切到512*512分辨率'''
def cutArray(l, num):
  avg = len(l) / float(num)
  o = []
  last = 0.0

  while last < len(l):
    o.append(l[int(last):int(last + avg)])
    last += avg

  return o
  
def convertjpg(jpgfile,outdir,width=512,height=512):
    img=Image.open(jpgfile)
    (l,h) = img.size
    rate = min(l,h) / width
    try:
        img = img.resize((int(l // rate),int(h // rate)),Image.BILINEAR)
        img = img.crop((0,0,width,height))
        img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

class thread(threading.Thread):
    def __init__(self, threadID, inpath, outpath, files):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.inpath = inpath
        self.outpath = outpath
        self.files = files
    def run(self):
        count = 0
        try:
            for file in self.files:
                convertjpg(self.inpath + file,self.outpath)
                count = count + 1
        except Exception as e:
            print(e)
        print('已处理图片数量：' +  str(count))
            
if __name__ == '__main__':
    inpath = './work/train/'
    outpath = './work/train/'
    files =  os.listdir(inpath)
    # for file in files:
    #     convertjpg(path + file,path)
    files = cutArray(files,8)
    T1 = thread(1, inpath, outpath, files[0])
    T2 = thread(2, inpath, outpath, files[1])
    T3 = thread(3, inpath, outpath, files[2])
    T4 = thread(4, inpath, outpath, files[3])
    T5 = thread(5, inpath, outpath, files[4])
    T6 = thread(6, inpath, outpath, files[5])
    T7 = thread(7, inpath, outpath, files[6])
    T8 = thread(8, inpath, outpath, files[7])
    
    T1.start()
    T2.start()
    T3.start()
    T4.start()
    T5.start()
    T6.start()
    T7.start()
    T8.start()
    
    T1.join()
    T2.join()
    T3.join()
    T4.join()
    T5.join()
    T6.join()
    T7.join()
    T8.join()
```

## 导入本项目所需的库

```python
import os
import cv2
import numpy as np
import paddle.dataset as dataset
from skimage import io,color,transform
import sklearn.neighbors as neighbors
import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import os
from skimage import io,color
import matplotlib.pyplot as plt
import six
```

## 定义数据预处理工具-DataReader

```python
'''准备数据，定义Reader()'''

PATH = 'work/train/'
TEST = 'work/train/'
Q = np.load('work/Q.npy')
Weight = np.load('work/Weight.npy')

class DataGenerater:
    def __init__(self):
        datalist = os.listdir(PATH)
        self.testlist = os.listdir(TEST)
        #datalist.sort()
        self.datalist = datalist
        

    def load(self, image):
        '''读取图片,并转为Lab，并提取出L和ab'''
        img = io.imread(image)
        lab = np.array(color.rgb2lab(img)).transpose()
        l = lab[:1,:,:]
        l = l.astype('float32')
        ab = lab[1:,:,:]
        ab = ab.astype('float32')
        # ab = (ab + 128)
        # ab = ab / 256
        return l,ab

    def mask(self,ls,abs,ws):
        return ls.astype('float32'),abs.astype('float32'),ws.astype('float32')

    def distribution(self,i):
        '''对Q空间进行统计，得到经验分布
        INPUT
            i   Q空间上的2维数组
        OUTPUT
            None
        '''
        nbrs = neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(Q)
        d, n = nbrs.kneighbors(i)
        w = np.array([])
        X = np.array([])
        Y = np.array([])
        for i in range(n.shape[0]):
            w = np.append(w,Weight[n[i][0]])
            x,y = self.lab2Q(d[i],n[i])
            X = np.append(X, x)
            Y = np.append(Y, y)
        w = w.reshape([1,256,256])
        X = X.reshape([1,256,256])
        Y = Y.reshape([1,256,256])
        q = np.zeros([2,256,256])
        q[0] = x
        q[1] = y
        return w,q

    def lab2Q(self,d,n):

        w = np.exp(-d ** 2 / (2 * 5 ** 2))
        w = w / np.sum(w)
        X = np.array([])
        Y = np.array([])
        for i in n:
            x = Q[i][0]
            y = Q[i][1]
            X = np.append(X,x)
            Y = np.append(Y,y)
        x = np.sum(X * w)
        y = np.sum(Y * w)
        return x,y

    def nd_to_2d(self,i, axis=0):
        '''将N维的矩阵转为2维矩阵
        INPUT
            i       N维矩阵
            axis    需要保留的维度
        OUTPUT
            o       转换的2维矩阵
        '''
        n = i.ndim
        shapeArray = np.array(i.shape)
        diff = np.setdiff1d(np.arange(0, n), np.array(axis))
        p = np.prod(shapeArray[diff])
        ax = np.concatenate((diff, np.array(axis).flatten()), axis=0)
        o = i.transpose((ax))
        o = o.reshape(p, shapeArray[axis])
        return o

    def _2d_to_nd(self,i,axis=1):

        a = i[:,:1].transpose().reshape([256,256])
        b = i[:,1:2].transpose().reshape([256,256])
        ab = np.zeros([2,256,256])
        ab[0] = a
        ab[1] = b
        return ab

    def weightArray(self,i):
        return self.distribution(self.nd_to_2d(i))


    def create_train_reader(self):
        '''给dataset定义reader'''

        def reader():
            for img in self.datalist:
                #print(img)
                try:
                    l, ab = self.load(PATH + img)
                    #print(ab)
                    yield l.astype('float32'), ab.astype('float32')
                except Exception as e:
                    print(e)

        return reader

    def create_test_reader(self,):
        '''给test定义reader'''

        def reader():
            for img in self.testlist:
                l,ab = self.load(TEST + img)
                yield l.astype('float32'),ab.astype('float32')

        return reader
def train(batch_sizes = 32):
    reader = DataGenerater().create_train_reader()
    return reader

def test():
    reader = DataGenerater().create_test_reader()
    return reader
```

## 定义网络功能模块并定义网络

我这个网络设计采用3组基本残差模块和2组反卷积层组成

```python
import IPython.display as display
import warnings
warnings.filterwarnings('ignore')

Q = np.load('work/Q.npy')
weight = np.load('work/Weight.npy')
Params_dirname = "work/model/gray2color.inference.model"

'''自定义损失函数'''
def createLoss(predict, truth):
    '''均方差'''
    loss1 = fluid.layers.square_error_cost(predict,truth)
    loss2 = fluid.layers.square_error_cost(predict,fluid.layers.fill_constant(shape=[BATCH_SIZE,2,512,512],value=fluid.layers.mean(predict),dtype='float32'))
    cost = fluid.layers.mean(loss1) + 26.7 / fluid.layers.mean(loss2)
    return cost

def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  bias_attr=True):
    tmp = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)
    return fluid.layers.batch_norm(input=tmp,act=act,momentum=0.5)


def shortcut(input, ch_in, ch_out, stride):
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_in, ch_out, stride):
    tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
    short = shortcut(input, ch_in, ch_out, stride)
    return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')


def layer_warp(block_func, input, ch_in, ch_out, count, stride):
    tmp = block_func(input, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1)
    return tmp

###反卷积层
def deconv(x, num_filters, filter_size=5, stride=2, dilation=1, padding=2, output_size=None, act=None):
    return fluid.layers.conv2d_transpose(
        input=x,
        num_filters=num_filters,
        # 滤波器数量
        output_size=output_size,
        # 输出图片大小
        filter_size=filter_size,
        # 滤波器大小
        stride=stride,
        # 步长
        dilation=dilation,
        # 膨胀比例大小
        padding=padding,
        use_cudnn=True,
        # 是否使用cudnn内核
        act=act
        # 激活函数
    )


def resnetImagenet(input):
    res1 = layer_warp(basicblock, input, 64, 128, 1, 2)
    res2 = layer_warp(basicblock, res1, 128, 256, 1, 2)
    res3 = layer_warp(basicblock, res2, 256, 512, 4, 1)
    deconv1 = deconv(res3, num_filters=313, filter_size=4, stride=2, padding=1)
    deconv2 = deconv(deconv1, num_filters=2, filter_size=4, stride=2, padding=1)
    return deconv2
```

## 训练网络

Tips:设置的超参数为：

- 学习率:2e-5
- Epoch:30
- Mini-Batch: 14
- 输入Tensor:[-1,1,512,512]

预训练的预测模型存放路径work/model/gray2color.inference.model

```python
BATCH_SIZE = 14
EPOCH_NUM = 30

def ResNettrain():
    gray = fluid.layers.data(name='gray', shape=[1, 512,512], dtype='float32')
    #w = fluid.layers.data(name='weight', shape=[1,256, 256], dtype='float32')
    truth = fluid.layers.data(name='truth', shape=[2, 512,512], dtype='float32')
    predict = resnetImagenet(gray)
    cost = createLoss(predict=predict,truth=truth)
    return predict,cost


'''optimizer函数'''
def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=2e-5,beta1=0.8)


train_reader = paddle.batch(paddle.reader.shuffle(
        reader=train(), buf_size=7500
    ),batch_size=BATCH_SIZE)
test_reader = paddle.batch(reader=test(), batch_size=10)

use_cuda = True
if not use_cuda:
    os.environ['CPU_NUM'] = str(6)
feed_order = ['gray', 'weight']
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

main_program = fluid.default_main_program()
star_program = fluid.default_startup_program()

'''网络训练'''
predict,cost = ResNettrain()

'''优化函数'''
optimizer = optimizer_program()
optimizer.minimize(cost)

exe = fluid.Executor(place)

def train_loop():
    gray = fluid.layers.data(name='gray', shape=[1, 512,512], dtype='float32')
    truth = fluid.layers.data(name='truth', shape=[2, 512,512], dtype='float32')
    feeder = fluid.DataFeeder(
        feed_list=['gray','truth'], place=place)
    exe.run(star_program)

    #增量训练
    fluid.io.load_persistables(exe, 'work/model/incremental/', main_program)
    
    for pass_id in range(EPOCH_NUM):
        step = 0
        for data in train_reader():
            loss = exe.run(main_program, feed=feeder.feed(data),fetch_list=[cost])
            step += 1
            if step % 100 == 0:
                try:
                    generated_img = exe.run(main_program, feed=feeder.feed(data),fetch_list=[predict])
                    plt.figure(figsize=(15,6))
                    plt.grid(False)
                    for i in range(10):
                        ab = generated_img[0][i]
                        l = data[i][0][0]
                        # l = np.zeros([256,256]) + 100
                        a = ab[0]
                        #a = a - 127
                        b = ab[1]
                        #b = b - 127
                        l = l[:, :, np.newaxis]
                        a = a[:, :, np.newaxis].astype('float64')
                        b = b[:, :, np.newaxis].astype('float64')
                        lab = np.concatenate((l, a, b), axis=2)
                        # img = (255 * np.clip(color.lab2rgb(lab), 0, 1)).astype('uint8')
                        img = color.lab2rgb((lab))
                        img = transform.rotate(img, 270)
                        img = np.fliplr(img)
                        plt.grid(False)
                        plt.subplot(2, 5, i + 1)
                        plt.imshow(img)
                        plt.axis('off')
                        plt.xticks([])
                        plt.yticks([])
                    msg = 'Epoch ID={0} Batch ID={1} Loss={2}'.format(pass_id, step, loss[0][0])
                    #print(msg)
                    plt.suptitle(msg,fontsize=20)
                    plt.draw()
                    plt.savefig('{}/{:04d}_{:04d}.png'.format('work/output_img', pass_id, step),bbox_inches='tight')
                    plt.pause(0.01)
                    display.clear_output(wait=True)
                except IOError:
                    print(IOError)
                
                fluid.io.save_persistables(exe,'work/model/incremental/',main_program)
                fluid.io.save_inference_model(Params_dirname, ["gray"],[predict], exe)
train_loop()
```

## 总结

这是我第一次通过循序渐进的方式叙述了项目的过程，我之前的基本上就放个代码。
对于训练结果我也emmmm...虽然本项目通过抑制平均化加大离散程度提高了着色的饱和度，但最终结果仍然有较大现实差距，只能对部分场景有比较好的结果，对人造场景（如超市景观等）仍然表现力不足。
接下来准备进一步去设计损失函数，目的是让网络着色结果足以欺骗人的”直觉感受“，而不是一味地接近真实场景。但是，马上就要开学了我也是.....（本项目无限鸽置...）