import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
import sys
import cv2,os,glob,datetime
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

#which batch and which epoch
num_round = int(sys.argv[1])
num_batch = int(sys.argv[2])
prefix = "model/ckpt-%d-0"%num_batch
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(0), numpy_batch_size=1)
mean_img = mx.nd.load("mean%d.bin"%num_batch)["mean_img"]

def oversample(images, crop_dims):

    im_shape = np.array(images.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    crops_ix = np.tile(crops_ix, (2, 1))

    # print crops_ix

    # Extract crops
    crops = np.empty((10, crop_dims[0], crop_dims[1],
                      im_shape[-1]), dtype=np.float32)
    ix = 0
    # for im in images:
    im = images
    # print im.shape
    for crop in crops_ix:
        crops[ix] = im[crop[0]:crop[2], crop[1]:crop[3], :]
        ix += 1
    crops[ix-5:ix] = crops[ix-5:ix, :, ::-1, :]
    # cv2.imshow('crop', crops[5,:,:,:])
    # cv2.waitKey()
    return crops



def PreprocessImage_oversample(path, show_img=False):
    # load image
    img = io.imread(path)
    # print("Original Image Shape: ", img.shape)
    rows, cols = img.shape[:2]
    if cols < rows: 
        resize_width = 320
        resize_height = resize_width * rows / cols;
    else:
        resize_height = 320
        resize_width = resize_height * cols / rows;

    resized_img = transform.resize(img, (resize_height, resize_width))

    if show_img:
        cv2.imshow("resized_img", resized_img)
        cv2.waitKey()

    batch = oversample(resized_img, (320,320))
    #batch = oversample(resized_img, (224,224))

    return batch


def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)



test_list = sorted(os.listdir('./test/'))

cnt = 0

X_test_id = []
Y_test = []

for f in test_list:
    cnt+=1
    X_test_id.append(f)
    batches = PreprocessImage_oversample('./test/%s'%(f))
    # print("batch Image Shape: ", batches.shape)

    probs = np.empty((10, 10), dtype=np.float32)
    idx = 0
    for batch in batches:
        sample = np.asarray(batch) * 256
        sample = np.swapaxes(sample, 0, 2)
        sample = np.swapaxes(sample, 1, 2)
        normed_img = sample - mean_img.asnumpy()
        #normed_img.resize(1, 3, 224, 224)
        normed_img.resize(1, 3, 320, 320)
        prob = model.predict(normed_img)[0]
        probs[idx] = prob
        idx += 1

    score = probs.mean(axis=0)
    a = np.array(score)
    Y_test.append(a.tolist())


    if cnt%100 == 1: print cnt,f

create_submission(Y_test,X_test_id,'vgg_10crop')
