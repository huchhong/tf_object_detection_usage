安装准备
--------

```
conda create -n tensorflow pip python=3.5
activate tensorflow
pip install pillow matplotlib pandas

git clone https://github.com/tensorflow/models.git
cd models/research/object_detecion
```
将xml_to_csv.py和generate_tfrecord.py拷贝到object_detection下

## 环境配置

每次进入终端后，都要设置一下PYTHONPATH：

```
windows: set PYTHONPATH=${install_dir}\models;${install_dir}\models\research;${install_dir}\models\research\slim
linux: export PYTHONPATH=${install_dir}/models:${install_dir}/models/research:${install_dir}/models/research/slim
```

## 编译protobuf

porotos目录下有很多proto文件，需要将它们用protoc转换为对应的python文件

### windows
```
conda install -c anaconda protobuf
protoc --python_out=. .\protos\train.proto ...
```
由于cmd终端不支持glob，因此需要手动输入protos目录下的所有proto

注意当前版本conda安装的最新protobuf是3.5版本，而当前tensorflow（1.10.0）要求protobuf的版本>=3.6.0，因此此时
tensorflow不可用。如果已经安装了tensorflow，需要将tensorflow,protobuf先卸载掉，然后重新安装tensorflow。
如果还未安装，也需要将protobuf卸载掉。如果没有以上问题，可以忽略。

### linux

TODO

目录说明
---------

samples/configs下有各种网络的配置文件，后面训练自己的数据的时候就从这里选择相应的网络的配置文件进行修改
```
mkdir training
mkdir images
mkdir images/train
mkdir images/test
```
数据准备
--------

用labelimg标注数据，将大部分数据放到images/train目录下，剩下的放到images/test目录下。

将xml转换成csv文件
```
python xml_to_csv.py
```

将csv文件转换成tf_record文件，转换之前需要修改generate_tfrecord.py，将id和标签名对应起来
```
def class_text_to_int(row_label):
    if row_label == 'GuoLiCheng':
        return 1
    elif row_label == 'maidongqingningkouwei':
        return 2
    elif row_label == 'meirishiliulvcha':
        return 3
    else:
        None
```
然后转换
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```

在training下生成labelmap.pbtxt，id和类名的对应需要和generate_tfrecord.py里的一致
```
item {
  id: 1
  name: 'GuoLiCheng'
}

item {
  id: 2
  name: 'maidongqingningkouwei'
}

item {
  id: 3
  name: 'meirishiliulvcha'
}
```

## 下载预训练的模型参数

到[detection_model_zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
下载预训练的模型参数，将其解压放到object_detection目录下


配置训练过程
------------

将samples/configs目录下想用的网络的config文件拷贝到training下（重命名），并做相应修改。

以ssd_mobilenet_v2_coco.config为例，需要修改的地方有：
```
num_classes： # 改为自己的数据集里的类别数
fine_tune_checkpoint： "PATH_TO_BE_CONFIGURED/model.ckpt"   # 下载好的预训练模型
num_steps： # 根据需要调整训练步数
train_input_reader：{
	tf_record_input_reader： {
		input_path： train.record  # 训练数据的tf_record文件
	}
	label_map_path："PATH/labelmap.pbtxt" # 生成的labelmap.pbtxt文件的路径
}

eval_input_reader: {
	tf_record_input_reader: {
		input_path: test.record  #测试数据的tf_record文件
	}
	label_map_path："PATH/labelmap.pbtxt"  # 生成的labelmap.pbtxt文件的路径
}

eval_config: {
  num_examples: 8000 # 测试集的样本数
  num_visualizations: 100 # tensorboard上可以看到的图片个数
  max_evals: 10  # 评估的次数。在训练的同时可以打开另一个评估进程。
}
```

开始训练
---------
```
python legacy/train.py --logtostderr --train_dir=training --pipeline_config_path=training/xxx.config
```
训练过程中可以用tensorboard查看训练的中间过程
```
tensorboard --logdir=training
```

同时也可以同步进行模型评估
```
python legacy/eval.py --logtostderr --pipeline_config_path=training/xxx.config --checkpoint_dir=training --eval_dir=eval
tensorboard --logdir=eval
```

训练结束后生成用于推断的模型
```
mkdir inference_graph
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/xxx.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```

评估
------

修改Object_detection_video.py或者Object_detection_webcam.py用视频或者摄像头测试模型的效果

或者查看测试集的检测效果
```
python legacy/eval.py --logtostderr --pipeline_config_path=training/xxx.config --checkpoint_dir=training --eval_dir=eval
tensorboard --logdir=eval
```
