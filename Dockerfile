FROM bvlc/caffe:cpu

ADD vessel_segmentation.py .
ADD requirements.txt .
ADD processing_vs processing_vs
ADD cnn_models cnn_models

ARG input_volume
ARG initialization
ARG type

ADD $input_volume input.nii.gz
ADD $initialization init.pkl

RUN pip install -r requirements,txt

CMD ["python", "vessel_segmentation.py input.nii.gz init.pkl ${type} segmentation/seg.nii.gz"]
