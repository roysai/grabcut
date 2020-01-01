import numpy as np
import pandas as pd
from skimage.io import imread, imshow, imsave
from skimage import img_as_ubyte
from skimage.transform import resize
import skimage
import cv2
import warnings
warnings.filterwarnings('ignore')

def read_image(url):
    try:
        img=imread(url)
        if img.ndim==2:
            img=img[:,:,np.newaxis]
            img=np.tile(img, (1,1,3))
        elif img.shape[2]==4:
            img=img[:,:,:3]
    except:
        print ('fail reading image {}'.format(url))
        img=np.zeros((224,224,3))
    return img


            
class GrabCutter():
    def __init__(self):
        pass
    
    def __call__(self, im_url, box, with_refine=False, brush_size=15, fg_color=(0, 255, 0), bg_color=(0, 0, 255)):
        """
        Parameters:
        im_url: url of image
        box: bounding box of image. Follow hydravision format (upper left X, upper left Y, lower left X, lower left Y)
        brush_size: brush size when choosing foreground and background
        fg_color: color of highlighted foreground
        bg_color: color of highlighted background
        
        return:
        image_with_box: array in cv2 image format, with bounding box drawn on it.
        segmented_img: segmented image after grabcut
        """
        
        self.brush_size=brush_size
        self.drawing=False
        self.foreground=False
        self.fg_color=fg_color
        self.background=False
        self.bg_color=bg_color
        if isinstance (im_url, str):
            self.image=read_image(im_url)
        else:
            self.image=im_url
        height, width, _=self.image.shape
        #limit the size
        if height>800:
            height=800
            width=int(800/self.image.shape[0]*width)
            self.image=resize(self.image, (height, width, _), preserve_range=True).astype(np.uint8)
            ratio=800/self.image.shape[0]
           # print ('transformed image size')
        elif width>800:
            width=800
            height=int(800/self.image.shape[1]*height)
            self.image=resize(self.image, (height, width, _), preserve_range=True).astype(np.uint8)
            ratio=800/self.image.shape[1]
            #print ('transformed image size')
        else:
            ratio=1
        #convert image BGR format
        self.original_img=self.image[:,:,::-1]
        #initialize bgdmodel and fgdmodel. Don't know what it is, but is needed for grabcut.
        self.bgdModel = np.zeros((1,65),np.float64)
        self.fgdModel = np.zeros((1,65),np.float64)
        
        #initialize mask
        self.mask=np.zeros(self.original_img.shape[:2],np.uint8)
        
        if box!='':
            box=box.split(',')
            ULX=int(int(box[0])/ratio)
            ULY=int(int(box[1])/ratio)
            LRX=int(int(box[2])/ratio)
            LRY=int(int(box[3])/ratio)
            self.rect=(ULX, ULY, LRX-ULX, LRY-ULY)
            self.img_with_box=cv2.rectangle(self.original_img, (ULX, ULY), (LRX,LRY), (0, 0, 255), 3)
        else:
            self.rect=(1,1, width-1, height-1)
            self.img_with_box=cv2.rectangle(self.original_img, 
                                            (self.rect[0], self.rect[1]), 
                                            (self.rect[1],self.rect[1]), 
                                            (0, 0, 255), 3)
       
        #perform initial grabcut with box
        #cv2.namedWindow('Original Image')
        #cv2.imshow('Original Image', self.original_img)
        cv2.grabCut(self.original_img, self.mask, self.rect, self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2=np.where((self.mask==2)|(self.mask==0), 0, 1).astype('uint8')
        self.segmented_img=self.original_img*mask2[:,:,np.newaxis]
        if with_refine:
            return self.img_with_box, self.refine_image()
        else:
            return self.img_with_box, self.segmented_img
     
    def select_foreground_background(self, event, x, y, flags, param):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.drawing=True
            if self.foreground:
                self.color=self.fg_color
                cv2.circle(self.segmented_img,(x,y),self.brush_size,self.color,-1)
                self.mask[y-self.brush_size:y+self.brush_size, x-self.brush_size:x+self.brush_size]=1
            elif self.background:
                self.color=self.bg_color
                cv2.circle(self.segmented_img,(x,y),self.brush_size,self.color,-1)
                self.mask[y-self.brush_size:y+self.brush_size, x-self.brush_size:x+self.brush_size]=0
        elif event==cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                if self.foreground:
                    cv2.circle(self.segmented_img,(x,y),self.brush_size,self.color,-1)
                    self.mask[y-self.brush_size:y+self.brush_size, x-self.brush_size:x+self.brush_size]=1
                elif self.background:
                    cv2.circle(self.segmented_img,(x,y),self.brush_size,self.color,-1)
                    self.mask[y-self.brush_size:y+self.brush_size, x-self.brush_size:x+self.brush_size]=0
        elif event==cv2.EVENT_LBUTTONUP:
            self.drawing=False
            
    def refine_image(self):
        """
        allow user to choose foreground and background
        
        useful keys:
        f: choose foreground. on the segmented image, click and drag on region of interest.
            Selected area will become green (default).
        b: choose background. on the segmented image, click and drag on region of interest.
            Selected area will become red (default).
        r: rerun grabcut iteration.
        spacebar: end grabcut iteration.
        =: increase brush size
        -: decrease brush size
        
        2 images will be generated: original image with bounding box and segmented image. Useful keys can only be 
        used on segmented image window.
        
        upon spacebar key (end grabcut iteration), final segmented image will be returned.
        """
        
        cv2.namedWindow('original image')
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.select_foreground_background)
        
        while True:
            cv2.imshow('image', self.segmented_img)
            cv2.imshow('original image', self.img_with_box)
            key=cv2.waitKey(1)
            
            if key==ord('f'):
                #print ('selecting foreground... ')
                self.foreground=True
                self.background=False
            elif key==ord('b'):
                #print ('selecting background... ')
                self.foreground=False
                self.background=True
            elif key==ord('r'): #reiterate grabcut with selected fore and background
                #print ('performing new grabCut...')
                self.background=False
                self.foreground=False
                _=cv2.grabCut(self.original_img, self.mask, None, self.bgdModel, self.fgdModel, 5, cv2.GC_INIT_WITH_MASK)
                mask2=np.where((self.mask==2) | (self.mask==0), 0, 1).astype('uint8')
                self.segmented_img=self.original_img*mask2[:,:, np.newaxis]
                #print ('grabcut iteration done!')
            elif key==ord('='): #increase brush size
                self.brush_size+=2
            elif key==ord('-'): #decrease brush size
                self.brush_size-=2
            
            elif key==ord (' '): #space bar to end
                #print ('I quit!')
                return self.segmented_img