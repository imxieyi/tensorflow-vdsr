
# coding: utf-8

# In[4]:

from PIL import Image
import os, sys, glob, re


# In[7]:

# Parameters
crop_width = 492
crop_height = 492
input_dir = 'E:\\OneDrive\\yande_dataset'
output_dir = './yande_small_small'


# In[10]:

img_list = glob.glob(os.path.join(input_dir, '*'))
img_list = [f for f in img_list if re.search('^\d+.png$', os.path.basename(f))]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[14]:

for path in img_list:
    print(path)
    image = Image.open(path)
    hoffset = 0
    voffset = 0
    hcrop = crop_width
    vcrop = crop_height
    if image.width > crop_width:
        hoffset = (image.width - crop_width) // 2
    elif image.width < crop_width:
        hcrop = image.width
    if image.height > crop_height:
        voffset = (image.height - crop_height) // 2
    elif image.height < crop_height:
        vcrop = image.height
    image = image.crop((hoffset, voffset, hoffset + hcrop, voffset + vcrop))
    image.save(path.replace(input_dir, output_dir, 1), 'PNG')


# In[ ]:



