import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization,concatenate,Conv2DTranspose,Dropout,AveragePooling2D,Add
from tensorflow.keras.models import Model
from Adaptive_Wing_Loss import Adaptive_Wing_Loss
#Spatial Configuration 
#Reference : PAYER, Christian, et al. Integrating spatial configuration into heatmap regression based CNNs for landmark localization. Medical image analysis, 2019, 54: 207-219.

def UNet(inputs):
  u = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
  u=BatchNormalization()(u)
  u = Conv2D(32,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u=Dropout(0.1)(u)
  u1 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u = AveragePooling2D(pool_size=(2, 2))(u1)

  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u=Dropout(0.2)(u)
  u2 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u = AveragePooling2D(pool_size=(2, 2))(u2)


  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u=Dropout(0.3)(u)
  u3 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u = AveragePooling2D(pool_size=(2, 2))(u3)

  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u=Dropout(0.3)(u)
  u4 = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u = AveragePooling2D(pool_size=(2, 2))(u4)

  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u) 
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u) 
  u=BatchNormalization()(u)
  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(128,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u) 
  u=BatchNormalization()(u)
  u = Conv2D(256,(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)  
  u=BatchNormalization()(u)


  u = tf.keras.layers.UpSampling2D(interpolation='bilinear')(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  
  u=Add()([u,u4])
  u = tf.keras.layers.UpSampling2D(interpolation='bilinear')(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)

  u=Add()([u,u3])
  u = tf.keras.layers.UpSampling2D(interpolation='bilinear')(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)

  u=Add()([u,u2])
  u = tf.keras.layers.UpSampling2D(interpolation='bilinear')(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)

  u=Add()([u,u1])
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)
  u=BatchNormalization()(u)
  u=Dropout(0.3)(u)
  u = Conv2D(128,3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(u)

  return u

def Spatial_Configuration(convc) :
  sconv = AveragePooling2D(pool_size=(2, 2))(convc)

  sconv = Conv2D(128,(7,7),  padding = 'same', kernel_initializer = 'he_normal')(sconv)
  sconv=BatchNormalization()(sconv)
  sconv=tf.keras.layers.LeakyReLU( alpha=0.1)(sconv)
  sconv=Dropout(0.25)(sconv)

  sconv = Conv2D(128,(7,7),  padding = 'same', kernel_initializer = 'he_normal')(sconv)
  sconv=BatchNormalization()(sconv)
  sconv=tf.keras.layers.LeakyReLU( alpha=0.1)(sconv)
  sconv=Dropout(0.25)(sconv)

  sconv = Conv2D(128,(7,7),  padding = 'same', kernel_initializer = 'he_normal')(sconv)
  sconv=BatchNormalization()(sconv)
  sconv=tf.keras.layers.LeakyReLU(alpha=0.1)(sconv)
  sconv=Dropout(0.25)(sconv)


  sconv = Conv2D(128,(7,7), padding = 'same', kernel_initializer = 'he_normal')(sconv)
  sconv=BatchNormalization()(sconv)
  sconv=tf.keras.layers.LeakyReLU( alpha=0.1)(sconv)
  sconv=Dropout(0.25)(sconv)

  sconv = Conv2D(128,(7,7), padding = 'same', kernel_initializer = 'he_normal')(sconv)
  sconv=BatchNormalization()(sconv)  
  sconv=tf.keras.layers.LeakyReLU( alpha=0.1)(sconv)
  sconv=Dropout(0.25)(sconv)

  sconv = tf.keras.layers.UpSampling2D(interpolation='bilinear')(sconv)
  sconv1 = Conv2D(6,(1,1), activation = 'tanh', padding = 'same', kernel_initializer =tf.compat.v1.truncated_normal_initializer(stddev=0.0001),kernel_regularizer=tf.keras.regularizers.l2(0.0005))(sconv)
  return sconv1

  


class SpatialConfigUnet():
    def __init__(self,shape=(256,256,1) ,optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),metrics="MSE"):
        self.shape=shape
        self.optimizer=optimizer
        self.metrics=metrics
    def __call__ (self):
        AWL=Adaptive_Wing_Loss()
        input = Input(shape = self.shape)
        unet_output=UNet(input)
        local_app = Conv2D(6,(1,1), activation = None, padding = 'same', kernel_initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.0001),kernel_regularizer=tf.keras.regularizers.l2(0.0005))(unet_output)
        config_output=Spatial_Configuration(unet_output)
        outputs = tf.keras.layers.Multiply()([ unet_output,config_output])
        model = tf.keras.Model(inputs = input, outputs = [local_app,outputs])
        model.compile(optimizer = self.optimizer, loss = AWL.Loss(), metrics = self.metrics)
        return model


