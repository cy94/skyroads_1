#!/bin/bash

TESTLIST="80.tif 81.tif 99.tif 100.tif 110.tif"
#TESTLIST="image.png"
#TESTLIST="new.jpg"

for i in $TESTLIST; do
    echo $i
    python create_detections.py -c ./models/vanilla.pb -o 'preds_output/'$i'.txt' '/data/xview/train_images/'$i
    #python create_detections.py -c ./models/vanilla.pb -o 'preds_output/'$i'.txt' $i
  #  python create_detections.py -c ./models/vanilla.pb -o 'preds_output/'$i'.txt' '/data/Varnasi/'$i
done
