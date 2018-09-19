# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

ENV http_proxy http://194.138.0.25:9400
ENV https_proxy https://194.138.0.25:9400
RUN export http_proxy=http://194.138.0.25:9400
RUN export https_proxy=https://194.138.0.25:9400
RUN export no_proxy="localhost,127.0.0.1,172.30.*,*.siemens.com,*.internal,*.local"


RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# Clone the Detectron repository
RUN git clone https://github.com/facebookresearch/detectron /detectron

# Install Python dependencies
RUN pip install -r /detectron/requirements.txt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Detectron root
WORKDIR /detectron

# Set up Python modules
RUN make

# [Optional] Build custom ops
RUN make ops
