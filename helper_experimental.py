import numpy as np
import scipy.misc
import ai_lab.dataloader as dataloader
import ai_lab.transform.trans2d_experimental as trans2d
from ai_lab.transform.augment2d import Augment2D
import cv2

dataset = None
aug = None

def init_dataset(path='./dataset',augment=Augment2D(blur=0.1,pixel_shift=0.2,rotate_hard=True,rotate_soft=True,noise=0.05,zoom=0.0)):
    global dataset,aug
    
    dataset = dataloader.load_dataset(
    path, \
    "IMAGE_FOLDER", \
    args={'rgb':True,'keep_mem':True})

    aug = augment

    print "DATASET-HYPER-PARAMS:",dataset.hyper_params
    print "Dataset:",dataset


def get_training_batch(batch_size,size=128,channels=3,color_jitter=True,validation=False):
    global dataset,aug
    assert dataset != None
    
    data,labels = dataset.get_batch(batch_size,validation=validation)
    data,labels = trans2d.as_numpy_array_of_shape(data,labels,size,size,channels)
    # Augment
    if validation == False:
        if aug == None:     aug = Augment2D(blur=0.1,pixel_shift=0.1,rotate_hard=True,rotate_soft=False,noise=0.0,zoom=0.0)
        data,labels = aug.augment(data,labels)

    data,labels = trans2d.normalize_uint8(data,labels)

    ### Augment training data here
    #if color_jitter:
    #    data = trans2d.color_jitter(data,min_value=-0.025,max_value=0.025)
    return data, labels

def get_validation_batch(batch_size,size=128,channels=3):
    return get_training_batch(batch_size,size=size,channels=channels,color_jitter=False,validation=True)

def get_sample_count():    
    return dataset.get_sample_count()

    
#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if len(images.shape)!=4 or images.shape[-1] != 3:
        images = np.reshape(images,(len(images),h,w,1))
    
        img = np.zeros((h * size[0], w * size[1],3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx / size[1]
            img[j*h:j*h+h, i*w:i*w+w,:] = np.concatenate([image,image,image],axis=2)
    else:

        img = np.zeros((h * size[0], w * size[1],3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx / size[1]
            img[j*h:j*h+h, i*w:i*w+w,:] = image

    return img
