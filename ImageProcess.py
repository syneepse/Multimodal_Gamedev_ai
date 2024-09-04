from PIL import Image
import os
import matplotlib.pylab as plt
from glob import glob
import numpy as np
import cv2

def splitImages(imgPath):
    import cv2,time
    img = cv2.imread(imgPath)
    img2 = img

    height, width, channels = img.shape
    # Number of pieces Horizontally 
    W_SIZE  = int(width/128)
    # Number of pieces Vertically to each Horizontal  
    H_SIZE = int(height/128)

    chunkDir = './imageChunks/'
    if not os.path.exists(chunkDir):
        os.makedirs(chunkDir)
    for ih in range(H_SIZE ):
        for iw in range(W_SIZE ):
        
            x = 128 * iw 
            y = 128 * ih
            h = (128)
            w = (128 )
            print(x,y,h,w)
            img = img[int(y):int(y+h), int(x):int(x+w)]
            #NAME = str(time.time()) 
            cv2.imwrite(chunkDir + str(ih)+str(iw) +  ".png",img)
            img = img2
    return chunkDir, W_SIZE, H_SIZE


def reSizeImg(uploaded_file):
    low_res_img = Image.open(uploaded_file,mode='r').convert("RGB")
    width, height = low_res_img.size
    correctWidth = getDiv128(width)
    correctHeight = getDiv128(height)
    correctSize = (correctWidth, correctHeight)

    resized_image = Image.new(low_res_img.mode, correctSize, (255,255,255)) 
    
    resized_image.paste(low_res_img, (0,0))

    
    resizedImgdir = './resizedImage'
    if not os.path.exists(resizedImgdir):
        os.makedirs(resizedImgdir)

    resized_image.save(resizedImgdir + '/newimg.png')
    return resizedImgdir + '/newimg.png'

    
def reconstruct(ImagesDir, numWidth, numHeight, chunkSize):
    fig, ax = plt.subplots(numHeight + 1, numWidth, figsize=(10, 10))   
    w = 0
    h = 0
    FinalImage = Image.new( mode="RGB",     size=(chunkSize*numWidth, chunkSize*numHeight), color=(255,255,255))
    for f in os.listdir(ImagesDir):
        if f.endswith('.png'):
            img_mpl = cv2.imread(ImagesDir + f)
            img_mpl = cv2.cvtColor(img_mpl, cv2.COLOR_BGR2RGB)

            Image_chunk = Image.open(ImagesDir + f,mode='r').convert("RGB")
             
            FinalImage.paste(Image_chunk, (w*chunkSize,h*chunkSize))


            ax[h][w].imshow(img_mpl)
            ax[h][w].axis('off')
            print(f, w, h)
            w = w + 1
            if w == numWidth:
                w = 0
                h = h + 1
    
    FinalImage.save('FinalImage.png')
    return 'FinalImage.png'

            
    plt.show()
            
    


    

def getDiv128(pixCount):
    for i in range(pixCount, pixCount+128):
        if i % 128 == 0:
            print(i)
            return i

