# TensorflowWorks



This repo includes tensorflow works. 


## DeptwiseConv3D and EfficientNet3D

Based  with optimization for tensorflow last version.

* https://github.com/qubvel/efficientnet
* https://github.com/ZFTurbo/efficientnet_3D  


#### Usage
* DepthWiseConv3D

```
from DepthwiseConv3D import DepthwiseConv3D
import tensorflow as tf
inputs = tf.keras.layers.Input(shape=(input_shape))
conv=DepthwiseConv3D(args)(inputs)
```
* EfficentNet3D

```
import tensorflow as tf
from EfficientNet3D import *
model=EfficientNet3DB3(input_shape=(32,32,32,3))
model.summary()
```

## StackedHourglassModel (Includes only model)

[The paper  - Stacked Hourglass Networks for Human Pose Estimation ](https://arxiv.org/abs/1603.06937)

Based  with optimization.
* https://github.com/yuanyuanli85/Stacked_Hourglass_Network_Keras

#### Usage

* StackedHourglassModel
```
import tensorflow as tf
from StackedHourglassModel import *
instance=StackedHourglass(num_classes, num_stacks, num_channels,inres,outres,bottleneck,loss,optimizer,metrics)
model=instance()
model.summary()
```

## Spatial Configuration into  U-Net

[ The paper - Integrating Spatial Configuration into Heatmap Regression Based CNNs for Landmark Localization](https://arxiv.org/pdf/1908.00748.pdf)

#### Usage

* Spatial Configuration into  U-Net
```
import tensorflow as tf
from SpatialConfigUnetCustom import *
instance=SpatialConfigUnet(shape,loss,optimizer,metrics,last_activation)
model=instance()
model.summary()
```

## Adaptive Wing Loss

[The paper - Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression](https://arxiv.org/abs/1904.07399)

#### Usage

* Adaptive Wing Loss
```
import tensorflow as tf
from Adaptive_Wing_Loss import *
AWL=Adaptive_Wing_Loss()
loss=AWL().Loss()
```
