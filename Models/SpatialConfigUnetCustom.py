import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization,concatenate,Conv2DTranspose,Dropout
from tensorflow.keras.models import Model

#CUSTOM CUSTOM BE CAREFULL
#Spatial Configuration 
#Reference : PAYER, Christian, et al. Integrating spatial configuration into heatmap regression based CNNs for landmark localization. Medical image analysis, 2019, 54: 207-219.

def UNet(inputs,last_activation):
    
    conv1 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    d1=Dropout(0.1)(conv1)
    conv2 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d1)
    b=BatchNormalization()(conv2)
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(b)
    conv3 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    d2=Dropout(0.2)(conv3)
    conv4 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d2)
    b1=BatchNormalization()(conv4)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(b1)
    conv5 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    d3=Dropout(0.3)(conv5)
    conv6 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d3)
    b2=BatchNormalization()(conv6)
    
    pool3 = MaxPooling2D(pool_size=(2, 2))(b2)
    conv7 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    d4=Dropout(0.4)(conv7)
    conv8 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d4)
    b3=BatchNormalization()(conv8)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(b3)
    conv9 = Conv2D(128,(3,3),activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    d5=Dropout(0.4)(conv9)
    conv10 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d5)
    b4=BatchNormalization()(conv10)
    
    
    conv11 = tf.keras.layers.UpSampling2D()(b4)
    x= concatenate([conv11,conv8])
    conv12 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    d6=Dropout(0.4)(conv12)
    conv13 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d6)
    b5=BatchNormalization()(conv13)
    
    
    conv14 = tf.keras.layers.UpSampling2D()(b5)
    x1=concatenate([conv14,conv6])
    conv15 = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x1)
    d7=Dropout(0.3)(conv15)
    conv16 = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d7)
    b6=BatchNormalization()(conv16)
    
    conv17 = tf.keras.layers.UpSampling2D()(b6)
    x2=concatenate([conv17,conv4])
    conv18 = Conv2D(64,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x2)
    d8=Dropout(0.2)(conv18)
    conv19 = Conv2D(64,(3,3) ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d8)
    b7=BatchNormalization()(conv19)
    
    conv20 = tf.keras.layers.UpSampling2D()(b7)
    x3=concatenate([conv20,conv2])
    conv21 = Conv2D(32,(3,3) ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x3)
    d9=Dropout(0.1)(conv21)
    conv22 = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(d9)
    
    
    convc = Conv2D(6,(1,1), activation = last_activation, padding = 'same', kernel_initializer = 'he_normal')(conv22)
    
    return convc

def Spatial_Configuration(convc,last_activation) :
  sconv = MaxPooling2D(pool_size=(2, 2))(convc)
  sconv = Conv2D(16,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sconv)
  sconv=Dropout(0.2)(sconv)
  sconv=BatchNormalization()(sconv)
  sconv = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sconv)
  sconv=Dropout(0.3)(sconv)
  sconv=BatchNormalization()(sconv)
  sconv = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sconv)
  sconv=Dropout(0.4)(sconv)
  sconv=BatchNormalization()(sconv)
  sconv = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(sconv)
  sconv=Dropout(0.4)(sconv)
  sconv=BatchNormalization()(sconv)
  sconv = Conv2DTranspose(16,(4,4), activation = 'relu', padding = 'same', strides=(2,2),kernel_initializer = 'he_normal')(sconv)
  sconv1 = Conv2D(6,(1,1), activation = last_activation, padding = 'same', kernel_initializer = 'he_normal')(sconv)
  return sconv1
      
  


class SpatialConfigUnet():
    def __init__(self,shape=(512,512,1) ,loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),metrics="accuracy",last_activation="sigmoid"):
        self.shape=shape
        self.loss=loss
        self.optimizer=optimizer
        self.metrics=metrics
        self.last_activation=last_activation
    def __call__ (self):
        input = Input(shape = self.shape)
        unet_output=UNet(input,self.last_activation)
        output2 = Conv2D(1,(1,1), activation = self.last_activation, padding = 'same', kernel_initializer = 'he_normal')(unet_output)
        config_output=Spatial_Configuration(unet_output,self.last_activation)
        outputs = tf.keras.layers.Multiply()([ unet_output,config_output])
        model = tf.keras.Model(inputs = input, outputs = [output2,outputs])
        model.compile(optimizer = self.optimizer, loss = self.loss, metrics = self.metrics)
        return model


