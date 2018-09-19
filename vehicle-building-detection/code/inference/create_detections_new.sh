#!/bin/bash

#TESTLIST="79.tif 180.tif"
#TESTLIST="image.png"
TESTLIST="image.jpg"

for i in $TESTLIST; do
    echo $i
    # python create_detections.py -c ./models/vanilla.pb -o 'preds_output/'$i'.txt' '/data/xview/train_images/'$i
    python create_detections_varnasi.py -c ./models/vanilla.pb -o 'preds_output/'$i'.txt' $i
  #  python create_detections.py -c ./models/vanilla.pb -o 'preds_output/'$i'.txt' '/data/Varnasi/'$i
done
