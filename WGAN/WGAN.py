import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Reshape,Input,Dense,Activation,Conv2D,Conv2DTranspose,BatchNormalization,concatenate,LeakyReLU,Flatten,AveragePooling2D,MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam,SGD,RMSprop
from keras.models import Sequential
from keras.utils import to_categorical
from keras import backend as k
import sys
import math
import argparse
def wasserstein_loss(y_label,y_predict):
    return -k.mean(y_label*y_predict)
def generator_model(inputs,img_size,channel,kernel_size=5,layer_kernel=[128,64,32]):
    layer_kernel.append(channel)
    img_resize=img_size//4
    x=inputs
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
    generator =Model(inputs,x,name='generator')
    return generator
def discriminator_model(inputs,img_size,channel,kernel_size=5,layer_kernel=[32,64,128,256]):
    x=inputs
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
    discriminator=Model(inputs,x,name='discriminator')
    return discriminator
def wgan(num_class,latent_dims,img_size,channel,kernel_size=5,g_layers=[128,64,32],d_layers=[32,64,128,256]):
    #Setting inputs
    z_inputs=Input(shape=(latent_dims,))
    #label_inputs=Input(shape=(num_class,))
    img_inputs=Input(shape=(img_size,img_size,channel))
    #Build D
    D=discriminator_model(img_inputs,img_size,channel,kernel_size=kernel_size,layer_kernel=d_layers)
    D.trainable=False
    #D.compile(loss=wassertein_loss,optimizer=RMSprop,metrics=['accuracy'])
    #Build G
    G=generator_model(z_inputs,img_size,channel,kernel_size=kernel_size,layer_kernel=g_layers)
    D.summary()
    G.summary()
    #Build CGAN
    #D.trainable=False
    WGAN=Model(z_inputs,D(G(z_inputs)),name='adversarial')
    WGAN.summary()
    models=D,G,WGAN
    return models
def train(models,data,params):
    import os
    from PIL import Image
    '''
    models = discriminator, generator, adversarial
    data = x_train, y_train
    params = optimizers, batch_size, latent_size, epochs, n_critic, clip_value, train_steps
    optimizers=d_opt,g_opt,a_opt
    '''
    d,g,a=models
    x_train=data
    optimizers,batch_size,latent_size,epochs,n_critic,clip_value,train_steps=params
    d_opt,g_opt,a_opt=optimizers
    img_size=x_train.shape[1]
    channel=x_train.shape[3]
    #Save history images
    save_interval=1
    save_dir=os.path.join(os.getcwd(),'history')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    #Compile Models
    #g.compile(loss=wasserstein_loss,optimizer=g_opt,metrics=['accuracy'])
    a.compile(loss=wasserstein_loss,optimizer=a_opt,metrics=['accuracy'])
    d.trainable=True
    d.compile(loss=wasserstein_loss,optimizer=d_opt,metrics=['accuracy'])
    #Training start
    epoch=0
    noise_input=np.random.uniform(-1.0,1.0,size=[100,latent_size])
    train_size=x_train.shape[0]
    real_labels=np.ones((batch_size,1))
    fake_labels=np.zeros((batch_size,1))
    while epoch<=epochs:
        epoch+=1
        index=0
        loss=0
        acc=0
        while index<=n_critic:
            index+=1
            rand_idx=np.random.randint(0,train_size,size=batch_size)
            real_imgs=x_train[rand_idx]
            noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
            fake_imgs=g.predict(noise,verbose=0)
            d.trainable=True
            real_loss,real_acc=d.train_on_batch(real_imgs,real_labels)
            fake_loss,fake_acc=d.train_on_batch(fake_imgs,-real_labels)
            loss+=0.5*(real_loss+fake_loss)
            acc +=0.5*(real_acc+fake_acc)
            for layer in d.layers:
                weights=layer.get_weights()
                weights=[np.clip(weight, -clip_value, clip_value) for weight in weights]
                layer.set_weights(weights)
        loss/=n_critic
        acc/=n_critic
        log="epoch: %d [discriminator loss:%6.3f, acc:%5.3f ]"%(epoch,loss,acc*100)
        noise=np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        d.trainable=False
        a_loss,a_acc=a.train_on_batch(noise,real_labels)
        log="%s [adversarial loss: %6.3f, acc:%5.3f ]"%(log,a_loss,a_acc)
        sys.stdout.write('\r')
        sys.stdout.write(log)
        sys.stdout.flush()
        if (epoch % save_interval) ==0:
            g.save_weights(g.name,True)
            d.save_weights(d.name,True)
        if (epoch % train_steps) ==1:
            generate_img=g.predict(noise_input)
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
#def wasserstein_loss(y_label,y_predict):
#    return -k.mean(y_label,y_predict)
if __name__=="__main__":
    #Get Arguements
    args=get_args()
    from keras.utils import plot_model
    from keras.datasets import cifar10,mnist
    (x_train,_),(_,_)=mnist.load_data()
    #Preprocessing data
    x_train=x_train.astype('float32')/255
    if len(x_train.shape)==3:
        x_train=np.expand_dims(x_train,axis=len(x_train.shape))
    if args.mode=="train":
        #Hyper parameters
        num_labels=10#y_train.shape[1]#len(np.unique(y_train))
        n_critic=5
        clip_value=0.01
        train_steps=10
        epochs=args.epochs
        batch_size=args.batch_size
        latent_dims=args.latent_dims
        img_size=x_train.shape[1]
        x_train=np.reshape(x_train,[-1,img_size,img_size,1])
        channel=x_train.shape[3]
        #Build Models
        d,g,a=wgan(num_labels,latent_dims,img_size,channel)
        plot_model(d,to_file='discriminator.png')
        plot_model(g,to_file='generator.png')
        plot_model(a,to_file='adversarial.png')
        models=d,g,a
        #Setting Optimizers
        d_opt = RMSprop(lr=0.0005)
        g_opt = RMSprop(lr=0.0005)
        a_opt = RMSprop(lr=0.0005)
        opts=d_opt,g_opt,a_opt
        #Setting params
        params=opts,batch_size,latent_dims,epochs,n_critic,clip_value, train_steps
        data=x_train
        models=d,g,a
        train(models,data,params)
    else:
        sys.exit()
