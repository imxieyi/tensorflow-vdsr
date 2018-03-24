im_raw = imread('426949.png');
yuv = rgb2ycbcr(im_raw);
y = im_raw(:,:,1);
imshow(y);