import numpy as np
import tensorflow as tf
from PIL import Image
Image.MAX_IMAGE_PIXELS=226836120 
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm
import argparse
from det_util import generate_detections
import os
import cv2 
from color import get_rgb
os.environ['CUDA_VISIBLE_DEVICES']='2'
"""
Inference script to generate a file of predictions given an input.

Args:
    checkpoint: A filepath to the exported pb (model) file.
        ie ("saved_model.pb")

    chip_size: An integer describing how large chips of test image should be

    input: A filepath to a single test chip
        ie ("1192.tif")

    output: A filepath where the script will save  its predictions
        ie ("predictions.txt")


Outputs:
    Writes a file specified by the 'output' parameter containing predictions for the model.
        Per-line format:  xmin ymin xmax ymax class_prediction score_prediction
        Note that the variable "num_preds" is dependent on the trained model 
        (default is 250, but other models have differing numbers of predictions)

"""

def chip_image(img, chip_size=(300,300)):
    """
    Segment an image into NxWxH chips

    Args:
        img : Array of image to be chipped
        chip_size : A list of (width,height) dimensions for chips

    Outputs:
        An ndarray of shape (N,W,H,3) where N is the number of chips,
            W is the width per chip, and H is the height per chip.

    """
    print (img.shape)
    width,height,_ = img.shape
    wn,hn = chip_size
    images = np.zeros((int(width/wn) * int(height/hn),wn,hn,3))
    k = 0
    for i in tqdm(range(int(width/wn))):
        for j in range(int(height/hn)):
            
            chip = img[wn*i:wn*(i+1),hn*j:hn*(j+1),:3]
            images[k]=chip
            
            k = k + 1
    
    return images.astype(np.uint8)

def draw_bboxes(img,boxes,classes):
    """
    Draw bounding boxes on top of an image

    Args:
        img : Array of image to be modified
        boxes: An (N,4) array of boxes to draw, where N is the number of boxes.
        classes: An (N,1) array of classes corresponding to each bounding box.

    Outputs:
        An array of the same shape as 'img' with bounding boxes
            and classes drawn

    """
    source = Image.fromarray(img)
    draw = ImageDraw.Draw(source)
    w2,h2 = (img.shape[0],img.shape[1])

    idx = 0

    for i in range(len(boxes)):
        xmin,ymin,xmax,ymax = boxes[i]
       # p1=(xmin,ymin)
       # p2=(xmax,ymax)
        c = classes[i]
        color = tuple(get_rgb(int(c),100))+(255,)
        draw.text((xmin+15,ymin+15), str(c))
       # cv2.rectangle(img, p1, p2,color, 1)


        for j in range(4):
                draw.rectangle(((xmin+j, ymin+j), (xmax+j, ymax+j)), outline=color)
    return source

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--checkpoint", default='pbs/model.pb', help="Path to saved model")
    parser.add_argument("-cs", "--chip_size", default=300, type=int, help="Size in pixels to chip input image")
    parser.add_argument("input", help="Path to test chip")
    parser.add_argument("-o","--output",default="predictions.txt",help="Filepath of desired output")
    args = parser.parse_args()

    #Parse and chip images
    arr = np.array(Image.open(args.input))
    chip_size = (args.chip_size,args.chip_size)
    images = chip_image(arr,chip_size)
    print(images.shape)

    #generate detections
    boxes, scores, classes = generate_detections(args.checkpoint,images)

    #Process boxes to be full-sized
    width,height,_ = arr.shape
    cwn,chn = (chip_size)
    wn,hn = (int(width/cwn),int(height/chn))

    num_preds = 250
    bfull = boxes[:wn*hn].reshape((wn,hn,num_preds,4))
    b2 = np.zeros(bfull.shape)
    b2[:,:,:,0] = bfull[:,:,:,1]
    b2[:,:,:,1] = bfull[:,:,:,0]
    b2[:,:,:,2] = bfull[:,:,:,3]
    b2[:,:,:,3] = bfull[:,:,:,2]

    bfull = b2
    bfull[:,:,:,0] *= cwn
    bfull[:,:,:,2] *= cwn
    bfull[:,:,:,1] *= chn
    bfull[:,:,:,3] *= chn
    for i in range(wn):
        for j in range(hn):
            bfull[i,j,:,0] += j*cwn
            bfull[i,j,:,2] += j*cwn
            
            bfull[i,j,:,1] += i*chn
            bfull[i,j,:,3] += i*chn
            
    bfull = bfull.reshape((hn*wn,num_preds,4))

    
    #only display boxes with confidence > .5
    bs = bfull[scores > 0.25]
    cs = classes[scores>0.25]
    s = args.input.split("/")[::-1]
    draw_bboxes(arr,bs,cs).save("p_bboxes/"+s[0].split(".")[0] + ".png")
    

    with open(args.output,'w') as f:
        for i in range(bfull.shape[0]):
            for j in range(bfull[i].shape[0]):
                #box should be xmin ymin xmax ymax
                box = bfull[i,j]
                class_prediction = classes[i,j]
                score_prediction = scores[i,j]
                f.write('%d %d %d %d %d %f \n' % \
                    (box[0],box[1],box[2],box[3],int(class_prediction),score_prediction))
