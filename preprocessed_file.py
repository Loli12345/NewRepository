import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance
from pylab import *
from skimage.morphology import watershed
from numpy import *
from astropy.visualization import (MinMaxInterval, SqrtStretch, ImageNormalize)
from scipy import ndimage 
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from skimage.filters import sobel
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import data
from skimage import restoration
from skimage import  img_as_float, color
from skimage.util import random_noise
from skimage import morphology
from numpy import *
im=Image.open('/Users/elhamalkabawi/NewDirectory/disc2/OAS1_0064_MR1/RAW/OAS1_0064_MR1_mpr-3_anon_sag_66.gif')
im = im.convert('L')
##########################################################
## denoising image
for i in range(2,im.size[0]-2):
    for j in range(2,im.size[1]-2):
        b=[]
        if im.getpixel((i,j))>0 and im.getpixel((i,j))<255:
            pass
        elif im.getpixel((i,j))==0 or im.getpixel((i,j))==255:
            c=0
            for p in range(i-1,i+2):
                for q in range(j-1,j+2):
                    if im.getpixel((p,q))==0 or im.getpixel((p,q))==255: 
                        c=c+1
            if c>6:
                c=0
                for p in range(i-2,i+3):
                    for q in range(j-2,j+3):
                        b.append(im.getpixel((p,q)))
                        if im.getpixel((p,q))==0 or im.getpixel((p,q))==255:
                            c=c+1
                if c==25:
                    a=sum(b)/25
                    #print a
                    im.putpixel((i,j),a)
                else:
                    p=[]
                    for t in b:
                        if t not in (0,255):
                            p.append(t)
                    p.sort()
                    im.putpixel((i,j),p[len(p)/2])
            else:
                b1=[]
                for p in range(i-1,i+2):
                    for q in range(j-1,j+2):
                        b1.append(im.getpixel((p,q)))
                im.putpixel((i,j),sum(b1)/9)
im.save("nonoise.jpg") 
plt.imshow(im)
im.show()
##########################################
im=array(im)
#edge dector
edges = canny(im/255.)
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('Canny detector')
imshow(edges)
show()
fill_coins = ndi.binary_fill_holes(edges)
#label_objects, nb_labels = ndi.label(fill_coins)
#sizes = np.bincount(label_objects.ravel())
#mask_sizes = sizes > 20
#mask_sizes[0] = 0
#coins_cleaned = mask_sizes[label_objects]
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(fill_coins, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('Filling the holes')
imshow(fill_coins)
show()
coins_cleaned = morphology.remove_small_objects(fill_coins, 21)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(coins_cleaned, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('Removing small objects')
imshow(coins_cleaned)
show()
######Region based segmentation
#region basedsegmentation
markers = np.zeros_like(im)
markers[im < 30] = 1
markers[im > 150] = 2
elevation_map = sobel(im)
segmentation = watershed(elevation_map, markers)
#edges = canny(elevation_map /255.)
segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndi.label(segmentation)

imshow(segmentation)
show()

#####normalization
def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(0):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        arr[...,i] *= (255.0/maxval)
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr
n=normalize(coins_cleaned)
imshow(n)
show()

