#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ==============================================================================
# Created for the MVA course: Object Recognition winter 2016.
# Project: lego assembly instructions
# ==============================================================================

"""Generate a new dataset by randomly combining modified version of an item with
a background.

The dataset is composed of the images plus annotation xml files following the
PASCAL VOC2012 style.

To run the script you have to :
    1. Move all background and label images in the following folders:
        - background:     '/input_images/background/'
        - item:           '/input_images/<label name>/'
    2. Indicate the number of images you want for each label and run the script
    using the command line. For instance to generate 1000 images of each label:
        python generate_data.py --n 1000
    3. All images are outputed in the folder '/output_images/'

Image supported format: *.jpg, *.png
"""

# import the necessary packages
import os
import sys
import cv2
import numpy as np
import logging
import uuid

# define the constant that we be used to generate the dataset
BGRD_COL = [0,0,0,255]  # background color blue green red alpha
MIN_SCALE = 0.5         # minimum value for scaling
MAX_SCALE = 2           # maximum value for scaling
PERSP_FRAC = 0.25       # maximal fraction for perspective distorsion
MAX_HUE = 20            # maximum uniform hue distorsion, range [0 179]
MAX_SAT = 20            # maximum uniform saturation distorsion, range [0 255]
MAX_VAL = 20            # maximum uniform value distorsion, range [0 255]
MAX_CONTRAST_DEV = 0.1  # maximum contrast deviation [1±MAX_CONTRAST_DEV]
MAX_BRIGHTNESS = 0.1    # maximum birghtness deviation [±255*MAX_BRIGHTNESS]
MIN_OFFSET = 30         # minimal margin to border, must be stricly positive

def _make_transparent(im):
    """Modify the alpha channel of the image to make all black pixel
    transparent.
    """
    # consider all pixel with background color as transparent
    msk = np.square(im[:,:]-BGRD_COL).sum(axis=2)
    im[msk==0,3] = 0

    # do some dilation erosion to have better background detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    im[:,:,3] = cv2.morphologyEx(im[:,:,3],cv2.MORPH_OPEN,kernel)
    im[:,:,3] = cv2.morphologyEx(im[:,:,3],cv2.MORPH_CLOSE,kernel)

    return im

def _random_flip(im):
    """Apply a random flip to an image.
    """
    # randomly generate the flip
    if np.random.randint(2):
        im = im[:,::-1,:]

    return im

def _random_rotation(im):
    """Apply a random rotation to an image and avoids cropping.
    """
    # define the rotation matrix
    rows,cols = im.shape[:2]
    angle = 360 * np.random.rand(1)[0]
    MRot = cv2.getRotationMatrix2D((int(cols / 2),int(rows / 2)),angle,1)

    # compute the new corner positions and the corresponding translation
    corners = np.array([[0,0,1],[cols-1,0,1],[0,rows-1,1],[cols-1,rows-1,1]])
    corners = MRot.dot(corners.T)
    tcol = 1-int(corners[0,:].min())
    trow = 1-int(corners[1,:].min())
    tcolp = max(0,tcol)        # positive part of tcol
    tcoln = min(0,tcol)        # negative part of tcol
    trowp = max(0,trow)        # positive part of trow
    trown = min(0,trow)        # negative part of trow

    # apply the pre-rotation translation
    MTrans = np.float32([[1,0,tcolp],[0,1,trowp]])
    im = cv2.warpAffine(im,MTrans,(cols + 2 * tcolp,rows + 2 * trowp))

    # apply the rotation after updating the rotation matrix
    rows,cols = im.shape[:2]
    MRot = cv2.getRotationMatrix2D((int(cols / 2),int(rows / 2)),angle,1)
    im = cv2.warpAffine(im,MRot,(cols,rows))

    # apply the post-rotation translation
    rows,cols = im.shape[:2]
    MTrans = np.float32([[1,0,tcoln],[0,1,trown]])
    im = cv2.warpAffine(im,MTrans,(cols + 2 * tcoln,rows + 2 * trown))

    return im

def _random_scale(im):
    """Apply a random scaling to an image.
    """
    # randomly generate the scale
    scale = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * np.random.rand(1)[0]

    # compensate for the perspective average downscaling
    scale = scale / (1 - PERSP_FRAC)

    # apply the scaling
    rows,cols = im.shape[:2]
    im = cv2.resize(im,(0,0),fx=scale,fy=scale)

    return im

def _random_perspective(im):
    """Apply a random perspective transformation to an image.
    """
    # compute geometric maximal distorsion
    rows,cols = im.shape[:2]
    rowDist = int(PERSP_FRAC * rows)
    colDist = int(PERSP_FRAC * cols)

    # create the two sets of points such that pts1 is mapped to pts2
    pts1 = np.float32([[0,0],[cols-1,0],[cols-1,rows-1],[0,rows-1]])
    corn1 = pts1[0,:]+[np.random.randint(colDist),np.random.randint(rowDist)]
    corn2 = pts1[1,:]+[-np.random.randint(colDist),np.random.randint(rowDist)]
    corn3 = pts1[2,:]+[-np.random.randint(colDist),-np.random.randint(rowDist)]
    corn4 = pts1[3,:]+[np.random.randint(colDist),-np.random.randint(rowDist)]
    pts2 = np.float32([corn1,corn2,corn3,corn4])

    # define the perspective matrix and apply it
    MPersp = cv2.getPerspectiveTransform(pts1,pts2)
    im = cv2.warpPerspective(im,MPersp,(cols,rows))

    # crop the image
    tcol = -min(pts2[0,0],pts2[3,0])
    trow = -min(pts2[0,1],pts2[1,1])
    cols = pts2[:,0].max()-pts2[:,0].min()
    rows = pts2[:,1].max()-pts2[:,1].min()
    MTrans = np.float32([[1,0,tcol],[0,1,trow]])
    im = cv2.warpAffine(im,MTrans,(cols,rows))
    
    return im

def _random_HSV(im):
    """Apply a random modification of the image in the HSV space.
    """
    # convert to the Hue-Saturation-Value colorspace
    imHSV = cv2.cvtColor(im[:,:,0:3],cv2.COLOR_BGR2HSV)
    h = imHSV[:,:,0].astype('int16')
    s = imHSV[:,:,1].astype('int16')
    v = imHSV[:,:,2].astype('int16')

    # compute the different uniform distorsion
    distHue = np.random.randint(-MAX_HUE,MAX_HUE)
    distSat = np.random.randint(-MAX_SAT,MAX_SAT)
    distVal = np.random.randint(-MAX_VAL,MAX_VAL)

    # apply the distorsion
    h = h + distHue
    s = s + distSat
    v = v + distVal

    # make sure that we are still in the right range
    h = np.mod(h,180)
    s[s<0] = 0
    v[v<0] = 0
    s[s>255] = 255
    v[v>255] = 255

    # convert back to Blue-Green-Red colorspace
    imHSV[:,:,0] = h[:,:].astype('uint8')
    imHSV[:,:,1] = s[:,:].astype('uint8')
    imHSV[:,:,2] = v[:,:].astype('uint8')
    imBGR = cv2.cvtColor(imHSV,cv2.COLOR_HSV2BGR)

    # reconstruct the alpha channel
    b,g,r = cv2.split(imBGR)
    alpha = im[:,:,3]
    b[alpha==0] = 0
    g[alpha==0] = 0
    r[alpha==0] = 0
    im = cv2.merge((b,g,r,im[:,:,3]))

    return im

def _random_contrast_brightness(im):
    """Apply a random modification of the contrast and brightness of the image.
    """
    # define the brightness and contrast parameter
    alpha = (1 - MAX_CONTRAST_DEV) + 2 * MAX_CONTRAST_DEV * np.random.rand(1)[0]
    beta = int(MAX_BRIGHTNESS * 255 * (- 1 + 2 * np.random.rand(1)[0]))

    # convert to a higher integer precision
    im = im.astype('int16')

    # apply the random contrast and brightness
    im[:,:,0:3] = alpha * im[:,:,0:3] + beta

    # make sure the values are still in range
    im[im[:,:,0]<0,0] = 0
    im[im[:,:,1]<0,1] = 0
    im[im[:,:,2]<0,2] = 0
    im[im[:,:,0]>255,0] = 255
    im[im[:,:,1]>255,1] = 255
    im[im[:,:,2]>255,2] = 255

    # apply alpha channel
    alpha = im[:,:,3]
    im[:,:,0][alpha==0] = 0
    im[:,:,1][alpha==0] = 0
    im[:,:,2][alpha==0] = 0

    # conert back to uint8
    im = im.astype('uint8')

    return im

def _final_crop(im):
    """Crop the image to conserve only the essential part.
    """
    # extract the alpha channel and build the projection on row and col
    alpha = im[:,:,3]
    alpha_col = alpha.sum(axis=0)
    alpha_row = alpha.sum(axis=1)

    # find the minimal and maximal positions
    col_min = np.argmax(alpha_col>0)
    row_min = np.argmax(alpha_row>0)
    col_max = np.argmax(alpha_col[::-1]>0)
    row_max = np.argmax(alpha_row[::-1]>0)

    # crop the image
    im = im[row_min:-row_max,col_min:-col_max,:]

    return im

def _item_background_merging(imItem,imBack,showsteps=False):
    """Merge the item image with the background using a random offset.
    """
    # apply all the random transformations to item image
    if showsteps: cv2.imshow('0 - original',imItem)
    imItem = _make_transparent(imItem)
    if showsteps: cv2.imshow('1 - transparent',imItem)
    imItem = _random_flip(imItem)
    if showsteps: cv2.imshow('2 - flip',imItem)
    imItem = _random_rotation(imItem)
    if showsteps: cv2.imshow('3 - rotation',imItem)
    imItem = _random_scale(imItem)
    if showsteps: cv2.imshow('4 - scaling',imItem)
    imItem = _random_perspective(imItem)
    if showsteps: cv2.imshow('5 - perspective',imItem)
    imItem = _random_HSV(imItem)
    if showsteps: cv2.imshow('6 - hsv',imItem)
    imItem = _random_contrast_brightness(imItem)
    if showsteps: cv2.imshow('7 - contrast and brightness',imItem)
    imItem = _final_crop(imItem)
    if showsteps: cv2.imshow('8 - final crop',imItem)


    # retrieve images dimensions and compute the corresponding translations
    rowsItem,colsItem = imItem.shape[:2]
    rowsBack,colsBack = imBack.shape[:2]
    # resize if the image is too big
    while (MIN_OFFSET>=rowsBack - rowsItem - MIN_OFFSET) or \
        (MIN_OFFSET>=colsBack - colsItem - MIN_OFFSET):
        imItem = cv2.resize(imItem,(0,0),fx=0.75,fy=0.75)
        rowsItem,colsItem = imItem.shape[:2]
    trow = np.random.randint(MIN_OFFSET,rowsBack - rowsItem - MIN_OFFSET)
    tcol = np.random.randint(MIN_OFFSET,colsBack - colsItem - MIN_OFFSET)
    
    # combine the two images
    alpha = imItem[:,:,3]
    imMerged = imBack.copy()
    imMerged[trow:trow + rowsItem,tcol:tcol + colsItem,0][alpha>0] = \
        imItem[:,:,0][alpha>0]
    imMerged[trow:trow + rowsItem,tcol:tcol + colsItem,1][alpha>0] = \
        imItem[:,:,1][alpha>0]
    imMerged[trow:trow + rowsItem,tcol:tcol + colsItem,2][alpha>0] = \
        imItem[:,:,2][alpha>0]

    # show the different steps
    if showsteps: 
        cv2.imshow('9 - merged',imMerged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # explicitely state the return variables for readibility
    width  = colsBack
    height = rowsBack
    depth  = 3
    xmin   = tcol
    ymin   = trow
    xmax   = tcol + colsItem - 1
    ymax   = trow + rowsItem - 1

    return imMerged,width,height,depth,xmin,ymin,xmax,ymax

def _get_files():
    """Retrieve the background and label name plus their images path."""

    # setup the logger
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')

    # check that the '/input_images' folder exists
    if not os.path.isdir("input_images"):
        logging.info("Directory '/input_images/' not found")
        sys.exit(1)
    
    # consider only directories that are not hidden
    # os.chdir("input_images")
    dump = os.listdir("input_images")
    dump = [l for l in dump if os.path.isdir("input_images/" + l)]
    labels_dir = [l for l in dump if l[0]!="."]

    # check that the '/input_images/background' folder exists
    if not os.path.isdir("input_images/background"):
        logging.info("Directory '/input_images/background'"
            " not found")
        sys.exit(1)

    # retrieve item directories and pictures and store them in a dictionnary
    imPaths = {}
    for label_dir in labels_dir:
        dump = os.listdir("input_images/" + label_dir)
        dump = [l for l in dump if l.lower().endswith(
            ('.png','.jpg','.jpeg'))]
        logging.info("%d images loaded for %s"%(len(dump),label_dir))
        imPaths[label_dir] = dump

    return imPaths


def _generate_xml(filename,width,height,depth,label,xmin,ymin,xmax,ymax):
    """ Generate the XML file corresponding to the randomly generated image.
    """
    xml = """<annotation>
        <folder>VOC2012</folder>
        <filename>%s</filename>
        <source>
            <database>MVA RecVis2016 Pierre and Leo</database>
            <annotation>Automatic generation</annotation>
            <image>Photographies</image>
        </source>
        <size>
            <width>%d</width>
            <height>%d</height>
            <depth>%d</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>%s</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>%d</xmin>
                <ymin>%d</ymin>
                <xmax>%d</xmax>
                <ymax>%d</ymax>
            </bndbox>
        </object>
    </annotation>"""%(filename,
        width,height,depth,label,xmin,ymin,xmax,ymax)

    return xml


def generate_data(n):
    """Generate n images with a random modification of the item image plus a 
    random selection of background.
    """

    # setup the logger
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    logging.info("Beginning dataset generation")

    # retrieve the images path
    imPaths = _get_files()

    # loop through all item class and generate the images
    for item in imPaths:
        if item == "background": continue

        # select the random selection of item and background
        rndmBack = np.random.randint(len(imPaths["background"]),size = n)
        rndmItem = np.random.randint(len(imPaths[item]),size = n)

        # loop through all selection
        for i in np.arange(0,n):
            # retrieve the path
            pathBack = imPaths["background"][rndmBack[i]]
            pathBack = os.path.join("input_images","background",pathBack)
            pathItem = imPaths[item][rndmItem[i]]
            pathItem = os.path.join("input_images",item,pathItem)

            # open the images and generate the new one
            imItem = cv2.imread(pathItem,-1)
            imBack = cv2.imread(pathBack,-1)
            imMerged = _item_background_merging(imItem,imBack)
            
            # create the output directory if it doesn't exists
            if not os.path.isdir("output_images"):
                os.mkdir("output_images")
                logging.info("Directory 'output_images' created")
            if not os.path.isdir("output_annotations"):
                os.mkdir("output_annotations")
                logging.info("Directory 'output_annotations' created")

            # generate an unique ID
            uID = str(uuid.uuid4())
            filename = uID

            # generate the xml
            xml = _generate_xml(filename,width,height,depth,class_label,
                xmin,ymin,xmax,ymax):

            # save the image
            cv2.imwrite(os.path.join("output_images",filename + ".jpg"),
                imMerged)
            logging.info("image saved as %s"%(filename + ".jpg"))
            
    return
    

if __name__ == "__main__":
    # define the number of images to generate and run the script
    n = 1
    generate_data(n)

