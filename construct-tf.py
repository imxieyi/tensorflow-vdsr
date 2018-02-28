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
    
    yuv_bicubic_out = bicubic_out.convert('YCbCr')
    raw_yuv_bicubic_out = np.ndarray((out_height, out_width, 3), 'u1', yuv_bicubic_out.tobytes())
    #raw_yuv_bicubic_out = np.array(yuv_bicubic_out.getdata())
    #raw_yuv_bicubic_out = raw_yuv_bicubic_out.reshape(out_height, out_width, len(yuv_bicubic_out.getbands()))
    out_bicubic_y = raw_yuv_bicubic_out[:,:,0:1].reshape(out_height, out_width)

    out_vdsr_y = sess.run([output_tensor], feed_dict={
                input_tensor: np.resize(out_bicubic_y, (1, out_bicubic_y.shape[0], out_bicubic_y.shape[1], 1))
            })
    out_vdsr_y = np.resize(out_vdsr_y, (out_bicubic_y.shape[0], out_bicubic_y.shape[1], 1))

    # To prevent black and white blocks on output image
    # https://en.wikipedia.org/wiki/YCbCr
    out_vdsr_y = out_vdsr_y.clip(16, 235)
    
    import copy
    out_vdsr_yuv = copy.deepcopy(raw_yuv_bicubic_out)

    out_vdsr_yuv[:,:,0:1] = out_vdsr_y

    out_image_vdsr = Image.fromarray(out_vdsr_yuv, 'YCbCr')
    out_image_vdsr = out_image_vdsr.convert('RGB')

    if args.o.endswith('png'):
        out_image_vdsr.save(args.o, 'PNG')
    elif args.o.endswith('jpg'):
        out_image_vdsr.save(args.o, 'JPEG')
    else:
        raise Exception('Invalid output file extension')
