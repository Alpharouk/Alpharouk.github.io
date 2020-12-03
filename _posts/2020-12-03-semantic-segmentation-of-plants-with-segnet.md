---
title: "Kaggle competition"
date: 2020-12-03
tags: [iameg segmentation, segnet, hdr images]
header:
  image: ""
excerpt: "Data Wrangling, Data Science, Messy Data"
mathjax: "true"
---

# Libraries needed for this notebook


```
import cv2
import imageio
from PIL import Image
import numpy as np
from keras.preprocessing.image import img_to_array
from keras import backend as K
from keras.layers import Layer
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import argparse
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
```

# Customized Maxpooling/Unpooling Layers

Reference: https://github.com/ykamikawa/tf-keras-SegNet


```
class MaxPoolingWithArgmax2D(Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=ksize,
            strides=strides,
            padding=padding)

        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = tf.shape(updates, out_type='int32')

            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3])

            ret = tf.scatter_nd(K.expand_dims(K.flatten(mask)),
                                  K.flatten(updates),
                                  [K.prod(output_shape)])

            input_shape = updates.shape
            out_shape = [-1,
                         input_shape[1] * self.size[0],
                         input_shape[2] * self.size[1],
                         input_shape[3]]
        return K.reshape(ret, out_shape)

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
                mask_shape[0],
                mask_shape[1]*self.size[0],
                mask_shape[2]*self.size[1],
                mask_shape[3]
                )
```

# Segnet model with some customized layers

reference : https://github.com/ykamikawa/tf-keras-SegNet


```
    
def segnet(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="softmax"):
    # encoder
    inputs = Input(shape=input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)


    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_6)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    
    
    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_8)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
   

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_11)
    print("Build enceder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    

    unpool_2 = MaxUnpooling2D(pool_size)([conv_14, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_17)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(128, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
   
    unpool_4 = MaxUnpooling2D(pool_size)([conv_21, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)

    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Reshape(
        (input_shape[0] * input_shape[1], n_labels),
        input_shape=(input_shape[0], input_shape[1], n_labels),
    )(conv_26)

    outputs = Activation(output_mode)(conv_26)
    print("Build decoder done..")

    model = Model(inputs=inputs, outputs=outputs, name="SegNet")

    return model


model = segnet((224,224,3), n_labels=5 ,kernel=3, pool_size=(2,2), output_mode="softmax")


```

    

# Data generator


```
def exr_to_jpg(path):
    im = imageio.imread(path)
    im_gamma_correct = np.clip(np.power(im, 0.45), 0, 1)
    im_fixed = Image.fromarray(np.uint8(im_gamma_correct*255))
    return im_fixed

def category_label(labels, dims, n_labels):
    x = np.zeros([dims[0], dims[1], n_labels])
    for i in range(dims[0]):
        for j in range(dims[1]):
            f=int(labels[i,j])
            x[i, j, f] = 1
    x = x.reshape(dims[0] * dims[1], n_labels)
    return x
def colorize(img):
    w=img.shape[0]
    h=img.shape[1]
    z=img.shape[2]
    l=np.zeros((w,h,3))
    for i in range(w):
        for j in range(h):
            if img[i,j,0]==1:
                l[i,j,0]=0
                l[i,j,1]=0
                l[i,j,2]=0
            elif img[i,j,1]==1:
                l[i,j,0]=255
                l[i,j,1]=0
                l[i,j,2]=0
            elif img[i,j,2]==1:
                l[i,j,0]=0
                l[i,j,1]=255
                l[i,j,2]=0
            elif img[i,j,3]==1:
                l[i,j,0]=0
                l[i,j,1]=0
                l[i,j,2]=255
            elif img[i,j,4]==1:
                l[i,j,0]=238
                l[i,j,1]=197
                l[i,j,2]=145
    return l

def class_pixels(img):
    w=img.shape[0]
    h=img.shape[1]
    z=img.shape[2]
    l=np.zeros((w,h,z))
    for i in range(w):
        for j in range(h):
            for f in range(z-1):
                if img[i,j,f]==np.max([img[i,j,0],img[i,j,1],img[i,j,2],img[i,j,3],img[i,j,4]]):
                    l[i,j,f]=1
    return l

def data_gen_small(img_dir,mask_dir,depth_dir,liste, batch_size, dims=(224,224), n_labels=5):
    while True:
        ix = np.random.choice(liste, batch_size)
        imgs = []
        labels = []
        for index in ix:
            # images
            img_path = img_dir[index]
            original_img = exr_to_jpg(img_path)
            array_img = img_to_array(original_img)/255
            imgs.append(array_img)
            
            # masks
            mask_path = mask_dir[index]
            original_mask=cv2.imread(mask_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            array_mask = category_label(original_mask[:, :, 0], dims, n_labels)
            labels.append(array_mask)
            
        imgs = np.array(imgs)
        labels = np.array(labels)
        yield imgs, labels
```

# Data preprocessing and splitting


```
import os
rgb=[]
depth=[]
mask=[]
node=[]

for dirs,subdir,files in os.walk('../input/synthetic-rgbd-images-of-plants/dataset of synthetic rgb-d plants/rgb_map'):
    for file_name in files:
        if file_name.endswith(".exr"):
            path_file=dirs+os.sep+file_name
            depth_file='../input/synthetic-rgbd-images-of-plants/dataset of synthetic rgb-d plants/depth_map/profondeur_map/'+file_name
            mask_file='../input/synthetic-rgbd-images-of-plants/dataset of synthetic rgb-d plants/semantic_map/segmentation2_map/'+file_name
            node_file='../input/synthetic-rgbd-images-of-plants/dataset of synthetic rgb-d plants/nodes_map/internoeuds_map/'+file_name
            rgb.append(path_file)
            depth.append(depth_file)
            mask.append(mask_file)
            node.append(node_file)
            
liste=np.arange(1,10000)
np.random.shuffle(liste)

train_list=liste[0:8000]
val_list=liste[8000:9000]
test_list=liste[9000:9999]


train_gen = data_gen_small(rgb
,mask,depth,liste=train_list,batch_size=16,dims=(224,224),n_labels=5)
val_gen=data_gen_small(rgb
,mask,depth,liste=val_list,batch_size=16,dims=(224,224),n_labels=5)
test_gen = data_gen_small(rgb
,mask,depth,liste=test_list,batch_size=1,dims=(224,224),n_labels=5)
```

# Metrics

Because the dataset contains imbalanced number of classes in each image, it's better to use a proper metric for that, which is Dice Coefficient.


```
def dice_coef(y_true, y_pred): 
    
    epsilon=1e-6
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return K.mean((numerator + epsilon) / (denominator + epsilon))

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
```


```
model.compile(loss=dice_coef_loss, optimizer='adam', metrics=["accuracy",dice_coef])
```

# Training

It took me while to train the model ( 24h on four external GPUs ). So will just write the code for training and load the trained model after.


```
history=model.fit_generator(
        train_gen,
        steps_per_epoch=500,
        epochs=50,
        validation_data=val_gen,
        validation_steps=62
    )
```

# Post-training

Here is the model trained on the same training set but externally in a server of 4 GPUs. Couldn't finish the training here because of the GPU quota and the successive shutdowns of the session.


```
model.load_weights('../input/pre-model-2/only_rgb_dice.hdf5')
```

# Training and validation curves

Once again I uploaded the history of my external training.


```
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# plotting of training and validation loss curves
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```


![png](/images/semantic-segmentation-of-plants-with-segnet_files/semantic-segmentation-of-plants-with-segnet_19_0.png)



![png](/images/semantic-segmentation-of-plants-with-segnet_files/semantic-segmentation-of-plants-with-segnet_19_1.png)


# ***Inference on test data***


```
h=0
for i in test_list:
    if h<50:
        
        img_path = rgb[i]
        original_img = exr_to_jpg(img_path)
            
        
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)    
        plt.imshow(original_img)
        array_img=img_to_array(original_img)/255
       
        array_img2 = np.reshape(array_img, (1,224,224,3))
        y_pred=model.predict(array_img2)
        y_pred=np.reshape(y_pred,(224,224,6))
        c=class_pixels(y_pred)
        o=colorize(c)
        plt.subplot(1,2,2)    
        plt.imshow(o)
        plt.show()
        h+=1
    else:
        break
```


![png](/images/semantic-segmentation-of-plants-with-segnet_files/semantic-segmentation-of-plants-with-segnet_21_0.png)



![png](/images/semantic-segmentation-of-plants-with-segnet_files/semantic-segmentation-of-plants-with-segnet_21_1.png)



![png](/images/semantic-segmentation-of-plants-with-segnet_files/semantic-segmentation-of-plants-with-segnet_21_2.png)



![png](/images/semantic-segmentation-of-plants-with-segnet_files/semantic-segmentation-of-plants-with-segnet_21_3.png)






```

```
