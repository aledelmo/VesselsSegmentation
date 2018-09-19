FROM bvlc/caffe:cpu

ADD vessel_segmentation.py .
ADD requirements.txt .
ADD processing_vs processing_vs
ADD cnn_models cnn_models

ARG input_volume
ARG initialization
ARG type
ARG output_volume

ADD $input_volume input.nii.gz
ADD $initialization init.pkl
ADD $output_volume output.nii.gz

RUN pip install -r requirements,txt

CMD ["python", "vessel_segmentation.py input.nii.gz init.pkl ${type} output.nii.gz"]
