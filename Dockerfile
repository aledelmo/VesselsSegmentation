FROM bvlc/caffe:cpu

ADD vessel_segmentation.py .
ADD requirements.txt .
ADD processing_vs processing_vs
ADD cnn_models cnn_models

ARG input_volume
ARG initialization
ARG type
ARG output_volume

RUN pip install -r requirements,txt

CMD ["python", "vessel_segmentation.py ${input_volume} ${initialization} ${type} ${output_volume}"]
