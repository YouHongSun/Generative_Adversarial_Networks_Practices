import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Reshape,Input,Dense,Activation,Conv2D,Conv2DTranspose,BatchNormalization,concatenate,LeakyReLU,Flatten,AveragePooling2D,MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.models import Sequential
from keras.utils import to_categorical
import sys
import math
import argparse
def generator_model(inputs,y_labels,img_size,channel,kernel_size=5,layer_kernel=[128,64,32]):
    layer_kernel.append(channel)
    img_resize=img_size//4
    x=concatenate([inputs,y_labels],axis=1)
    x=Dense(img_resize*img_resize*channel)(x)
    x=Reshape((img_resize,img_resize,channel))(x)
    for kernel in layer_kernel:
        if kernel > layer_kernel[-2]:
            strides=2
        else:
            strides=1
        x=BatchNormalization()(x)
        x=Activation('relu')(x)
        x=Conv2DTranspose(filters=kernel,kernel_size=kernel_size,strides=strides,padding='same')(x)
    x=Activation('sigmoid')(x)
    generator =Model([inputs,y_labels],x,name='generator')
    return generator
def discriminator_model(inputs,y_labels,img_size,channel,kernel_size=5,layer_kernel=[32,64,128,256]):
    x=inputs
    y=Dense(img_size*img_size*channel)(y_labels)
    y=Reshape((img_size,img_size,channel))(y)
    x=concatenate([x,y])
    for kernel in layer_kernel:
        if kernel == layer_kernel[-1]:
            strides=1
        else:
            strides=2
        x=LeakyReLU(alpha=0.2)(x)
        x=Conv2D(filters=kernel,
                kernel_size=kernel_size,
                strides=strides,
                padding='same')(x)
    x=Flatten()(x)
    x=Dense(1)(x)
    x=Activation('sigmoid')(x)
    discriminator=Model([inputs,y_labels],x,name='discriminator')
    return discriminator
def cgan(num_class,latent_dims,img_size,channel,kernel_size=5,g_layers=[128,64,32],d_layers=[32,64,128,256]):
    #Setting inputs
    z_inputs=Input(shape=(latent_dims,))
    label_inputs=Input(shape=(num_class,))
    img_inputs=Input(shape=(img_size,img_size,channel))
    #Build D
    D=discriminator_model(img_inputs,label_inputs,img_size,channel,kernel_size=kernel_size,layer_kernel=d_layers)
    #Build G
    G=generator_model(z_inputs,label_inputs,img_size,channel,kernel_size=kernel_size,layer_kernel=g_layers)
    D.summary()
    G.summary()
    #Build CGAN
    D.trainable=False
    CGAN=Model([z_inputs,label_inputs],D([G([z_inputs,label_inputs]),label_inputs]),name='adversarial')
    CGAN.summary()
    #Compile Models
    #G.compile(loss='binary_crossentropy',optimizer=Adam)
    #CGAN.compile(loss='binary_crossentropy',optimizer=Adam)
    #D.trainable=True
    #D.compile(loss='binary_crossentropy',optimizer=Adam)
    models=D,G,CGAN
    return models
def train(models,data,params):
    import os
    from PIL import Image
    '''
    models = discriminator, generator, adversarial
    data = x_train, y_train
    params = optimizers, batch_size, latent_size, epochs
    optimizers=d_opt,g_opt,a_opt
    '''
    d,g,a=models
    x_train,y_train=data
    optimizers,batch_size,latent_size,epochs=params
    d_opt,g_opt,a_opt=optimizers
    img_size=x_train.shape[1]
    channel=x_train.shape[3]
    num_class=y_train.shape[1]
    d_opt,g_opt,a_opt=optimizers
    #Save history images
    save_interval=1
    save_dir=os.path.join(os.getcwd(),'history')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    #Compile Models
    g.compile(loss='binary_crossentropy',optimizer=g_opt)
    a.compile(loss='binary_crossentropy',optimizer=a_opt)
    d.trainable=True
    d.compile(loss='binary_crossentropy',optimizer=d_opt)
    #Training start
    epoch=0
    noise_input=np.random.uniform(-1.0,1.0,size=[100,latent_size])
    noise_class=np.eye(num_class)[np.arange(100)%num_class]
    #print(noise_class)
    while epoch<=epochs:
        epoch+=1
        index=0
        while index<=int(x_train.shape[0]/batch_size):
            index+=1
            rand_idx=np.random.randint(0,x_train.shape[0],size=batch_size)
            real_img=x_train[rand_idx]
            real_label=y_train[rand_idx]
            noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
            fake_label=np.eye(num_class)[np.random.choice(num_class,batch_size)]
            fake_img=g.predict([noise,fake_label],verbose=0)
            x=np.concatenate((real_img,fake_img))
            labels=np.concatenate((real_label,fake_label))
            y=[1]*batch_size+[0]*batch_size
            d_loss=d.train_on_batch([x,labels],y)
            noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
            y=np.ones([batch_size,1])
            d.trainable=False
            g_loss=a.train_on_batch([noise,fake_label],y)
            d.trainable=True
            idx=int(index/x_train.shape[0]*batch_size*50)
            sys.stdout.write('\r')
            sys.stdout.write("epoch:%3d, batch:%d/%d [%-50s]%d%% d_loss:%5.3f, g_loss:%5.3f"%(epoch,index-1,int(x_train.shape[0]/batch_size),'='*idx,2*idx,d_loss,g_loss))
            sys.stdout.flush()
        g.save_weights(g.name,True)
        d.save_weights(d.name,True)
        generate_img=g.predict([noise_input,noise_class])
        generate_img=combine_images(generate_img)
        generate_img *=255
        Image.fromarray(generate_img.astype(np.uint8)).save(save_dir+'/'+str(epoch)+'.png')
def combine_images(generated_images):
    num = generated_images.shape[0]
    channel=generated_images.shape[len(generated_images.shape)-1]
    #print(channel)
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]
    return image
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--latent_dims", type=int, default=100)
    args = parser.parse_args()
    return args
if __name__=="__main__":
    #Get Arguements
    args=get_args()
    from keras.utils import plot_model
    from keras.datasets import cifar10,mnist
    (x_train,y_train),(_,_)=mnist.load_data()
    #Preprocessing data
    x_train=x_train.astype('float32')/255
    y_train=to_categorical(y_train)
    if len(x_train.shape)==3:
        x_train=np.expand_dims(x_train,axis=len(x_train.shape))
    if args.mode=="train":
        #Hyper parameters
        num_labels=y_train.shape[1]#len(np.unique(y_train))
        #print('num_labels',num_labels)
        epochs=args.epochs
        batch_size=args.batch_size
        latent_dims=args.latent_dims
        img_size=x_train.shape[1]
        channel=x_train.shape[3]
        #Build Models
        d,g,a=cgan(num_labels,latent_dims,img_size,channel)
        plot_model(d,to_file='discriminator.png')
        plot_model(g,to_file='generator.png')
        plot_model(a,to_file='adversarial.png')
        models=d,g,a
        #Setting Optimizers
        d_opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        a_opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        opts=d_opt,g_opt,a_opt
        #Setting params
        params=opts,batch_size,latent_dims,epochs
        data=x_train,y_train
        models=d,g,a
        train(models,data,params)
    else:
        sys.exit()
