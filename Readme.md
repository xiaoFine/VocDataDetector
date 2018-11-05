## 利用Tensorflow Object Detection API创建自定义的目标检测模型

### 0. 任务概览
任务是基于flicker提供的一万张图片，训练出一个能够是识别出包括人、汽车等在内的20种类别的物体，并标志在途中相对应的位置。
>目标检测（Object Detection）与图片分类、分割（Segmentation）同属于图像理解的三个任务，目前主流的模型有两阶段和单阶段法，其区别在于前者大多将检测与分类分成两个阶段学习，而后者则多是将分类与检测相统一的框架。


### 1. 牛刀小试
YOLO（You Look Only Once)模型是典型的单阶段目标识别模型，有着众多的分支和演化版本。在其[官网](https://pjreddie.com/darknet/yolo/)中给出了基于VOC Data的预训练（pre-trained）模型，能够识别人、车、狗等常见物体。我在Ubuntu上通过加载预训练模型的权重文件测试了一张图片，结果较为理想。以下是是官网指导：
- 首先是安装Darknet
```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```
- 下载权重文件
```
wget https://pjreddie.com/media/files/yolov3.weights
```
- 开始测试
```
./darknet detect cfg/yolov3.cfg yolov3.weights image1.jpg
```
- 结果
```
... ...
  104 conv    256  3 x 3 / 1    76 x  76 x 128   ->    76 x  76 x 256  3.407 BFLOPs
  105 conv    255  1 x 1 / 1    76 x  76 x 256   ->    76 x  76 x 255  0.754 BFLOPs
  106 yolo
Loading weights from ../yolov3.weights...Done!
dog: 100%
dog: 98%
person: 100%
person: 100%
person: 81%
```
![](http://wx3.sinaimg.cn/mw690/afb7b7d8ly1fwx9rtituoj20rs0ij41p.jpg)


### 2. 自定义模型
为了快速的厘清目前检测模型的学习流程，我采用了tensorflow开源的目标检测API，利用预定义的模型配置来训练网络。

> 所谓预定义的模型，是指通过键值对的形式固化成.config的模型配置，包括但不限于卷积结构、batch_size、正则参数等，其允许自定义（[Configuring the Object Detection Training Pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)）和下载/加载样本配置（[Sample Configs](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)），具体参考官方文档:[Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

#### 1) 数据处理
数据官网已经将一万张图片均分成训练集和测试集，其各自目录结构如下：
```
--VOCdevkit
    |-Annotations
    |-ImageSets
    |-JPEGImages
    |-SegmentationClass
    |-SegmentationObject
```
其中我主要使用了图片源文件JPEGImages和图片标签Annotations（.xml文件)。每个xml文件记录了图片信息：

```
<annotation>
	<folder>VOC2007</folder>
	<filename>000005.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>325991873</flickrid>
	</source>
	<owner>
		<flickrid>archintent louisville</flickrid>
		<name>?</name>
	</owner>
	<size>
		<width>500</width>
		<height>375</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>chair</name>
		<pose>Rear</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>263</xmin>
			<ymin>211</ymin>
			<xmax>324</xmax>
			<ymax>339</ymax>
		</bndbox>
	</object>
    ... ...
    ... ...
```

其中我们需要的是图片大小(width,height,depth默认为3可以不要)、待检测目标（name,bndbox)，当然也需要图片文件名。

由于大量图片不适合存储在内存中，tf提供了TFRecord类型的数据接口，因此我们需要做的是把.xml结构化数据转换成tfrecord。
>官方：tf.data API 支持多种文件格式，因此您可以处理那些不适合存储在内存中的大型数据集。例如，TFRecord 文件格式是一种面向记录的简单二进制格式，很多 TensorFlow 应用采用此格式来训练数据

此处我参考了开源项目[Raccoon Detector Dataset](https://github.com/datitran/raccoon_dataset)中的两个脚本，先将.xml转换成.csv，后通过tf.python_io.TFRecordWriter得到tfrecord，其中每一个图片记录如上文所述，即
```
'filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'
```
之后运行脚本，分别生成训练集train.record和测试集的test.record。
最后需要写一个label map, 按照20种类逐一编号即可，注意编号要与xml2csv中一致:
```
item{
    id:1
    name:"aeroplane"
}
item{
    id:2
    name:"bicycle"
}
item{
    id:3
    name:"bird"
}
... ...
```
#### 2) 试用接口
在Github上clone整个[tensorflow/model](https://github.com/tensorflow/models)，按照官方指导安装依赖、编译protobuf（建议使用3.4.0版本）。
另外官方提供了一个[jupyter notebook](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)脚本供用户调用、测试模型，在环境下run all cell即可得到两个测试样例的检测结果。或是按照官方测试方法:
```
python object_detection/builders/model_builder_test.py
```

如果Windows下抛出 No module named 'object_detection' 错误，参考[Issue](https://github.com/tensorflow/models/issues/2031#issuecomment-353693018)：
>1. model/research目录下执行
    ```
    python setup.py build
    ```
    ```
    python setup.py install
    ```
>2. 在model/research/slim目录下执行
    ```
    pip install -e .
    ```


notebook中默认使用的是一个SSD模型，它也是一个单阶段模型，预测速度较快，精度一般。
>SSD（[Single Shot Multibox Detector](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46448-0.pdf))，较YOLO而言，SSD能更好的检测到小物体，因其基于[VGG](https://arxiv.org/abs/1409.1556)得到的不同尺度的feature map；另外SSD有更多的anchor box（貌似是4:2)

#### 3）定义模型
我没有从零搭建模型，而是借助了官方提供的SSD模型。下载[模型](ssd_mobilenet_v1_coco_2018_01_28.tar.gz)（主要是pipeline）和[配置文件](https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v1_coco.config)。

按照需求修改模型配置：
```
model {
  ssd {
    num_classes: 20
...
 fine_tune_checkpoint: "../ssd_mobilenet_v1_coco_2018_01_28/model.ckpt"
···
train_input_reader: {
  tf_record_input_reader {
    input_path: "data/train.record"
  }
  label_map_path: "data/voc_label_map.pbtxt"
}
...
eval_input_reader: {
  tf_record_input_reader {
    input_path: "data/test.record"
  }
  label_map_path: "data/voc_label_map.pbtxt"
  shuffle: false
  num_readers: 1
}
...
```
最后整理我们的文件结构如下：
```
voc_detection_project
    |-data
        |-train.record
        |-test.record
        |-voc_label_map.pbtxt
    |-images
        |-000001.jpg
        |-000002.jpg
        |-...
    |-ssd_mobilenet_v1_coco.config
    |-ssd_mobilenet_v1_coco_2018_01_28
        |-pipeline.config
        |-...
    |-training
```
其中data和image存放了数据和标签，ssd_mobilenet_v1_coco.config是模型配置，training用来存放checkpoint。

#### 4）训练模型
调用/legacy/train.py脚本进行训练（注意相对路径）：
```
python train.py \
--logtostderr \
--train_dir=../training \
--pipeline_config_path=../training/ssd_mobilenet_v1_coco.config
``` 
可以通过tensorboard来查看训练过程：
```
tensorboard --logdir==\training --host=127.0.0.1
```
![tb](http://wx3.sinaimg.cn/mw690/afb7b7d8ly1fwxhzfr24cj216h0df3z6.jpg)
> 遇到 "tensorboard refuse to connect"之类的问题，可能是logdir问题，需要指向包含events.out.tfevents的文件夹路径
    ```
    --logdir==training:path\to\training
    ```

> 用低压cpu+集显的surface pro跑了4个checkpoint之后挂了，后来借了1080Ti跑的，速度相差100倍。。。

#### 5)测试模型
官方自带了模型导出脚本，选择checkpoint和输出路径即可：
```
python export_inference_graph.py 
--input_type image_tensor \
--pipeline_config_path training/ssd_mobilenet_v1_coco.config \
--trained_checkpoint_prefix training/model.ckpt-123 --output_directory voc_graph
```
运行在voc_graph文件夹下得到frozen_inference_graph.pb，通过官方的测试脚本加载模型，并选取一张测试集中的图片：
```
MODEL_NAME = 'voc_graph'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'voc_label_map.pbtxt')
...
#方便起见我选了一张图放到官方的测试集目录下
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '00003{}.jpg'.format(i)) for i in range(1,3)]
...
```
测试结果：
- ckpt-123的结果（笔记本跑的，只迭代了4次），total loss接近7。样例表明模型基本无法使用。。。
>![123-1](http://wx2.sinaimg.cn/mw690/afb7b7d8ly1fwxjejexemj209u0d1n3t.jpg)
>![123-2](http://wx3.sinaimg.cn/mw690/afb7b7d8ly1fwxjennyd3j20jt0bf454.jpg)
>![123-3](http://wx2.sinaimg.cn/mw690/afb7b7d8ly1fwxjer1gh9j20hp0d1n4i.jpg)
>![123-4](http://wx3.sinaimg.cn/mw690/afb7b7d8ly1fwxjeum6gpj209w0d147j.jpg)

- ckpt-5971（1080Ti跑的，截止目前还在训练）
>![5971-1](http://wx1.sinaimg.cn/mw690/afb7b7d8ly1fwxjjmurkrj209u0d1gs5.jpg)
![5971-2](http://wx2.sinaimg.cn/mw690/afb7b7d8ly1fwxjjq8avuj20jt0bf7ai.jpg)
![5971-3](http://wx1.sinaimg.cn/mw690/afb7b7d8ly1fwxjjtf5n7j20hp0d1n46.jpg)
![5971-4](http://wx2.sinaimg.cn/mw690/afb7b7d8ly1fwxjjxn3uhj209w0d147j.jpg)
效果还可以，等待完全训练完的结果
_____
### 3. 关键原理
以YOLO为例，有些方法是YOLO独有的，有些则是大多数检测模型都有的。
- 网格划分
通过将网格划分成N*N，每个网格Gird将拥有一个（1+4+C）长的特征向量，其中1标识网格内是否有物体，4个位置（与上述不同的是YOLO描述位置采用的是中心点-相对网格左上角坐标+长宽的方式），C是类别数量。根据物体所在边界框（即bndbox）与网格关系，有三种情况：
    1. 物体所在的边界框只在一个网格内，则该物体边界属于该网格；
    2. 物体所在边界框覆盖了多个网格。这种情况最常见，YOLO的做法是有边界框的中心点所在网格决定整个物体的所属网格。
    3. 一个网格内有多个物体的边界框。此处就引入锚点框（Anchor box）的概念
- 锚点框
YOLOv2（论文起的是YOLO9000）给每个网格分配了2个额外的“框”，意味着每个网格能够同时预测两个物体。
锚点框的形状是提前定义的，且两个互不相同。
则现在的分配策略是比较物体的边界框与网格的锚点框的相似性，物体最终分配到最相似的网格的那个锚点框。
同时网格的特征向量变成了(1+4+C)*2,2是锚点框的数量。
- 相似性
为了描述锚点框和边界框的相似性，通常的做法是利用IoU（Intersection over Union），一图以蔽之：
![iou](https://pic2.zhimg.com/80/v2-316f0ffd2d0b0fed3c206bd7616e9edd_hd.jpg)
- 非极大抑制
非极大抑制（Non-max Suppression)简单而言是为了在预测时找出置信度比较高的边界框，即抑制冗余的框。对某一类别而言，其步骤如下：
    1. 对所有框的置信度排序，得到置信度最高的边界框M并保留
    2. 遍历剩余的框，删去其中与M的IoU大于阈值的框
    3. 在剩余框中重复步骤1-2，最终得到每次保留的边界框
对其他类别重复上述步骤即可。
- 过滤
在进行NMS之前，由于YOLO模型对每个网格都产生了2个预测框，所以需要通过类别阈值（class score）对并不包含物体的网格进行过滤。

以上即我认为对目标检测较为重要的技术。

### 参考资料
>1. [目标检测](https://zhuanlan.zhihu.com/p/34142321)
>2. [Training Custom Object Detector](https://pythonprogramming.net/training-custom-objects-tensorflow-object-detection-api-tutorial/)
>3. [YOLO官网](https://pjreddie.com/darknet/yolo/)
>4. [YOLO论文](https://arxiv.org/abs/1506.02640)
>5. [YOLOv2](https://arxiv.org/abs/1612.08242)
>6. [SSD论文](https://arxiv.org/abs/1512.02325)