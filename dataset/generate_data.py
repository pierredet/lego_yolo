# Created for the MVA course: Object Recognition winter 2016.
# Project: lego assembly instructions
# ==============================================================================

"""Generate a new dataset by randomly combining modified version of an item with
a background.

The dataset is composed of the image plus an annotation xml file following the
PASCAL VOC2012 style.
"""

# import the necessary packages
import cv2
import numpy as np

# define the constant that we be used to generate the dataset
BGRD_COL = [0,0,0,255]	# background color blue green red alpha
MIN_SCALE = 0.5			# minimum value for scaling
MAX_SCALE = 1.5 		# maximum value for scaling
PERSP_FRAC = 0.25 		# maximal fraction for perspective distorsion
MAX_HUE = 20 			# maximum uniform hue distorsion, range [0 179]
MAX_SAT = 20			# maximum uniform saturation distorsion, range [0 255]
MAX_VAL = 20			# maximum uniform value distorsion, range [0 255]

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
	tcolp = max(0,tcol)		# positive part of tcol
	tcoln = min(0,tcol)		# negative part of tcol
	trowp = max(0,trow)		# positive part of trow
	trown = min(0,trow)		# negative part of trow

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
	im = cv2.warpPerspective(im, MPersp, (cols,rows))

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
	imHSV = cv2.cvtColor(im[:,:,0:3], cv2.COLOR_BGR2HSV)
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
	imBGR = cv2.cvtColor(imHSV, cv2.COLOR_HSV2BGR)

	# reconstruct the alpha channel
	b,g,r = cv2.split(imBGR)
	alpha = im[:,:,3]
	b[alpha==0] = 0
	g[alpha==0] = 0
	r[alpha==0] = 0
	im = cv2.merge((b,g,r,im[:,:,3]))

	return im

# # define the offset of the overlay and pad the item with zeros
# x 	   = np.random.randint(imBack.shape[0]-imItem.shape[0])
# y 	   = np.random.randint(imBack.shape[1]-imItem.shape[1])
# npad   = ((x,imBack.shape[0]-imItem.shape[0]-x),
# 		 (y,imBack.shape[1]-imItem.shape[1]-y),(0,0))
# imItem = np.pad(imItem, pad_width=npad, mode='constant', constant_values=0)

# # add item to the background
# imBack[imItem.sum(axis=2)!=0,:] = imItem[imItem.sum(axis=2)!=0,:]


if __name__ == "__main__":
	# define both image path and open them
	pathItem = "item.png"
	pathBack = "background.jpg"
	imItem = cv2.imread(pathItem,-1)
	imBack = cv2.imread(pathBack,-1)


	# apply all the transformations
	imItemTrans = _make_transparent(imItem)
	imItemFlip = _random_flip(imItemTrans)
	imItemRot = _random_rotation(imItemFlip)
	imItemScl = _random_scale(imItemRot)
	imItemPersp = _random_perspective(imItemScl)
	imHSV = _random_HSV(imItemPersp)

	# show the different steps
	cv2.imshow('0 - original',imItem)
	cv2.imshow('1 - transparent',imItemTrans)
	cv2.imshow('2 - flip',imItemFlip)
	cv2.imshow('3 - rotation',imItemRot)
	cv2.imshow('4 - scaling',imItemScl)
	cv2.imshow('5 - perspective',imItemPersp)
	cv2.imshow('6 - hsv',imHSV)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


