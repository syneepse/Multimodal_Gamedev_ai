import Reranker as rr
import shapey
import SD_Upscale
import SD_text2image as t2i
import SD_image2image as sdi2i

models = [ 'shape-e', 'SD3-t2i', '4xUpscale', 'SD3-i2i' ]
def txttoImg(prompt):
    print("stable Diffusion made")
    returnImage = 'Sample-Image.jpg'
    returnImage = t2i.runModel(prompt)
    return returnImage

def txtto3d(prompt):
    print("3d Object")
    returnObject = 'example_mesh_0.obj'
    returnObject = shapey.getShape(prompt)
    return returnObject


def imgUpscale(inputString, inputImage):
    print("Upscale4x")
    returnImage = 'Sample-Image.jpg'
    returnImage = SD_Upscale.generate(inputImage, inputString)
    return returnImage

def imgCreate(inputString, inputImage):
    print("img creator")
    #returnImage = 'Sample-Image.jpg'
    returnImage = sdi2i.runModel(inputImage, inputString)
    return returnImage
    



def ImagePrompt(inputDict):
    print("If there is an image and text in the prompt, this will be displayed")
    modelID = rr.correctPromptImage(inputDict['text']) + 2
    print(models[modelID])
    if(modelID == 2):
        print('upscale')
        outimg = imgUpscale(inputDict['text'], inputDict['files'][0])
    else:
        outimg = imgCreate(inputDict['text'], inputDict['files'][0])    
    #outimg = "D:\Development\Intel Project\DevTool\Sample-Image.jpg"
    return models[modelID], True ,outimg,None
    # return modelID

def TextPrompt(inputString):
    print("if there is only a text prompt, this will be displayed")
    modelID = rr.correctPromptText(inputString)
    print(models[modelID])
    imgbool = True
    outmodel = None
    outimg = None
    if(modelID == 0):
        imgbool = False
        #outimg = None
        outmodel = txtto3d(prompt=inputString)

    else:
        imgbool = True
        outimg = txttoImg(prompt=inputString)

    #imgbool = True
    #outimg = "D:\Development\Intel Project\DevTool\Sample-Image.jpg"
    #outmodel = None
    # if modelID == 0:
    #     imgbool = False
    #     outimg = None
    #     outmodel = 'example_mesh_0.ply'
        
    return models[modelID], imgbool, outimg,outmodel
    # return modelID

def PromptInput(string):
    if len(string['files']) == 0:
        return TextPrompt(string['text'])
    else:
        return ImagePrompt(string)
