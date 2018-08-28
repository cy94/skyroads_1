import cv2
from ctypes import *
import math
import random
import os
import numpy as np
from timeit import time

colors = [(255,255,255), (0,255,255), (0,255,255), (0,255,0), (238,232,170), (221,34,54)]

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./darknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
#get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr
def load_net_meta():
    net = load_net("cfg/yolov3.cfg", "cfg/yolov3.weights", 0)
    meta = load_meta("cfg/coco.data")
    return net, meta
def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum,0)
    #dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def detect_sumit(net, meta, im, thresh=.5, hier_thresh=.5, nms=.45):
    t1 = time.time()
    #im = load_image(image, 0, 0)
    if isinstance(im, bytes):
        # image is a filename
        # i.e. image = b'/darknet/data/dog.jpg'
        im = load_image(im, 0, 0)
    else:
        # image is an nparray
        # i.e. image = cv2.imread('/darknet/data/dog.jpg')
        im, nd_image = array_to_image(im)
        rgbgr_image(im)
    fps = (1./(time.time()-t1))
    #print("fps taken to convert numpy in image instance :", fps)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    if isinstance(im, bytes): free_image(im)
    free_detections(dets, num)
    return res
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    #net = load_net("cfg/tiny-yolo.cfg","cfg/tiny-yolo.weights", 0)
    net = load_net(b"yolov3-traffic.cfg",b"/mnt/backup/yolov3-traffic_30000.weights",0)
    meta = load_meta(b"traffic_new_2.data")
    r = detect(net, meta, b"/mnt/image.png")
    frame = cv2.imread("/mnt/image.png")
    print("hggsdj")
    print(len(r))
    print(r)
    for result in r:
        label = result[0]
        probability = result[1]
        bbox = result[2]
        color = colors[np.random.choice(range(len(colors)))]
        bbox1 = ( int(bbox[0]-bbox[2]/2), int(bbox[1]-bbox[3]/2), int(bbox[2]), int(bbox[3]))
        #cv.circle(img = frame, center= (bbox[0],bbox[1]), radius = 5, color = colors[0],thickness = -1 )
        #cv.circle(img = frame, center= (bbox[0]+bbox[2],bbox[1],bbox[3]), radius = 5, color = colors[0],thickness = -1 )

        # print("debug ", type(bbox[0]))
        cv2.rectangle(frame,(bbox1[0], bbox1[1]), (bbox1[0]+bbox1[2], bbox1[1]+ bbox1[3]), color, 3)
        #cv2.putText(img = frame, text = label,org =  (bbox[0], bbox[1]),fontScale = 2, fontFace = cv.FONT_HERSHEY_SIMPLEX, color = (255,255,0), thickness = 2)
        #cv2.putText(img = frame, text = label,org =  (int(bbox[0]), int(bbox[1])),fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 2,  color = (255,255,0), thickness = 2)
        #cv.putText(frame,label,(bbox[0], bbox[1]), cv.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv.LINE_AA)
        cv2.putText(frame, label.decode('unicode_escape'),(int(bbox1[0]), int(bbox1[1])), cv2.FONT_HERSHEY_SIMPLEX, 5e-3 * 200, (0,0,0),2)
    cv2.imwrite("/mnt/result.png",frame)

	
