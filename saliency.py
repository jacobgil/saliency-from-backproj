import cv2, sys
import numpy as np

def backproject(source, target, levels = 2, scale = 1):
	hsv = cv2.cvtColor(source,	cv2.COLOR_BGR2HSV)
	hsvt = cv2.cvtColor(target,	cv2.COLOR_BGR2HSV)
	# calculating object histogram
	roihist = cv2.calcHist([hsv],[0, 1], None, [levels, levels], [0, 180, 0, 256] )

	# normalize histogram and apply backprojection
	cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
	dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256], scale)
	return dst

def saliency_by_backprojection(img):
	cv2.pyrMeanShiftFiltering(img, 2, 10, img, 4)

	backproj = np.uint8(backproject(img, img, levels = 2))
	cv2.normalize(backproj,backproj,0,255,cv2.NORM_MINMAX)
	saliencies = [backproj, backproj, backproj]
	saliency = cv2.merge(saliencies)

	cv2.pyrMeanShiftFiltering(saliency, 20, 200, saliency, 2)
	saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)
	cv2.equalizeHist(saliency, saliency)

	return 255-saliency

def saliency_map(img):
	saliency_hsv = saliency_by_backprojection(img * 1, backproject)
	saliency = saliency_hsv
	(T, saliency) = cv2.threshold(saliency, 200, 255, cv2.THRESH_BINARY)
	return saliency

def largest_contours_rect(saliency):
	contours, hierarchy = cv2.findContours(saliency * 1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key = cv2.contourArea)
	return cv2.boundingRect(contours[-1])

def refine_saliency_with_grabcut(img, saliency):
	rect = largest_contours_rect(saliency)
	bgdmodel = np.zeros((1, 65),np.float64)
	fgdmodel = np.zeros((1, 65),np.float64)
	saliency[np.where(saliency > 0)] = cv2.GC_FGD
	mask = saliency
	cv2.grabCut(img, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
	mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	return mask

def backprojection_saliency(img):
	saliency = saliency_map(img)
	mask = refine_saliency_with_grabcut(img, saliency)
	return mask

if __name__ == "__main__":
	name = sys.argv[1].strip(".jpg")

	img = cv2.imread(sys.argv[1], 1)
	img = cv2.resize(img, (640/2, 480/2))
	mask = backprojection_saliency(img)
	segmentation = img*mask[:,:,np.newaxis]

	cv2.imshow("original", img)
	cv2.imshow("segmentation", segmentation)
	cv2.waitKey(-1)
