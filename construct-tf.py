# coding: utf-8

import argparse

parser = argparse.ArgumentParser(
    description='Scale image using given VDSR model.')
parser.add_argument('input', type=str, help='input image')
parser.add_argument('-m', type=str, help='VDSR model(checkpoint)', required=True)
parser.add_argument('-o', type=str, help='output image', required=True)
parser.add_argument('-s', type=int, help='scale factor', choices=[2, 3, 4], required=True)
args = parser.parse_args()

import numpy as np
from scipy import misc
from PIL import Image
import glob, os, re
from PSNR import psnr
import scipy.io
import pickle
from MODEL import model
#from MODEL_FACTORIZED import model_factorized
import time
import tensorflow as tf

# parameters
scale_factor = args.s

input_image = Image.open(args.input)
(width, height) = input_image.size
out_width = width * scale_factor // 1
out_height = height * scale_factor // 1
print('Size:', width, height)
print('Out Size:', out_width, out_height)

with tf.Session() as sess:
    input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
    shared_model = tf.make_template('shared_model', model)
    output_tensor, weights     = shared_model(input_tensor)
    saver = tf.train.Saver(weights)
    tf.global_variables_initializer().run()
    saver.restore(sess, args.m)

    bicubic_out = input_image.resize((width*scale_factor, height*scale_factor), Image.BICUBIC)

    raw_bicubic_out = np.array(bicubic_out.getdata())

    raw_bicubic_out = raw_bicubic_out.reshape(out_height, out_width, len(bicubic_out.getbands()))

    out_bicubic_rgb = raw_bicubic_out[:,:,0:3]
    out_bicubic_a = raw_bicubic_out[:,:,3]

    # RGB -> YUV
    # http://www.pythonexample.com/snippet/python/rgb2yuv_yuv2rgbpy_quasimondo_python
    def RGB2YUV(rgb):
        m = np.array([[ 0.29900, -0.16874,  0.50000],
                     [0.58700, -0.33126, -0.41869],
                     [ 0.11400, 0.50000, -0.08131]])
        yuv = np.dot(rgb, m)
        yuv[:,:,1:] += 128.0
        return yuv
    out_bicubic_yuv = RGB2YUV(out_bicubic_rgb)
    out_bicubic_y = out_bicubic_yuv[:,:,0:1].reshape(out_height, out_width)

    out_vdsr_y = sess.run([output_tensor], feed_dict={
                input_tensor: np.resize(out_bicubic_y, (1, out_bicubic_y.shape[0], out_bicubic_y.shape[1], 1))
            })
    out_vdsr_y = np.resize(out_vdsr_y, (out_bicubic_y.shape[0], out_bicubic_y.shape[1], 1))

    import copy
    out_vdsr_yuv = copy.deepcopy(out_bicubic_yuv)

    out_vdsr_yuv[:,:,0:1] = out_vdsr_y

    # YUV -> RGB
    def YUV2RGB( yuv ):
        m = np.array([[ 1.0, 1.0, 1.0],
                     [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                     [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
        rgb = np.dot(yuv,m)
        rgb[:,:,0]-=179.45477266423404
        rgb[:,:,1]+=135.45870971679688
        rgb[:,:,2]-=226.8183044444304
        return rgb.clip(0, 255).astype('uint8')
    
    out_vdsr_rgb = YUV2RGB(out_vdsr_yuv)

    out_image_vdsr = Image.fromarray(out_vdsr_rgb, 'RGB')

    if args.o.endswith('png'):
        out_image_vdsr.save(args.o, 'PNG')
    elif args.o.endswith('jpg'):
        out_image_vdsr.save(args.o, 'JPEG')
    else:
        raise Exception('Invalid output file extension')
