import mxnet as mx
import logging
import numpy as np
from skimage import io, transform
import sys

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

#which batch and which epoch
num_round = int(sys.argv[1])
num_batch = int(sys.argv[2])
prefix = "model/ckpt-%d-0"%num_batch
#change mx.gpu(0) to mx.gpu(1) if you want to use the other GPU card
model = mx.model.FeedForward.load(prefix, num_round, ctx=mx.gpu(0), numpy_batch_size=1)
mean_img = mx.nd.load("mean%d.bin"%num_batch)["mean_img"]

def PreprocessImage(path, show_img=False):
    # load image
    img = io.imread(path)
    #print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (320, 320))
    if show_img:
        io.imshow(resized_img)
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256
    # swap axes to make image from (224, 224, 4) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean 
    normed_img = sample - mean_img.asnumpy()
    normed_img.resize(1, 3, 320, 320)
    return normed_img

dict_explain = {
    0: 'safe driving',
    1: 'texting - right',
    2: 'talking on the phone - right',
    3: 'texting - left',
    4: 'talking on the phone - left',
    5: 'operating the radio',
    6: 'drinking',
    7: 'reaching behind',
    8: 'hair and makeup',
    9: 'talking to passenger'
}

import os
test_list = sorted(os.listdir('./test/'))
fw = open('submission_vgg_batch_%d_epoch_%d.csv'%(num_batch, num_round),'w')
fw.write('img,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9\n')
fw_exam = open('exam_vgg_%d_%d.txt'%(num_batch, num_round),'w')
cnt = 0
for f in test_list:
    cnt+=1
    batch = PreprocessImage('./test/%s'%(f))
    prob = model.predict(batch)[0]
    output_prob = ['%.6f'%p for p in prob]
    fw.write('%s,%s\n'%(f,','.join(output_prob)))
    # Argsort, get prediction index from largest prob to lowest
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = pred[0]
    #print("Top1: ", top1, dict_explain[top1])
    fw_exam.write('%s,%d,%s\n'%(f,top1,dict_explain[top1]))
    if cnt%1000 == 1: print cnt,f,top1,dict_explain[top1]

fw_exam.close()
fw.close()
