FROM bvlc/caffe:cpu

WORKDIR .
#VOLUME /test_docker

ARG folder

ADD vessel_segmentation.py .
ADD requirements.txt .
ADD processing_vs processing_vs
ADD cnn_models cnn_models
ADD test_docker.py .

ADD test_dataset/coroT2Cube.nii.gz .
ADD test_dataset/init.pkl .

#ARG input_volume
#ARG initialization
#ARG type

#ADD $input_volume input.nii.gz
#ADD $initialization init.pkl

#RUN ls
RUN pip install -r requirements.txt

RUN ls
#RUN ls /test_docker

#CMD ["python", "vessel_segmentation.py", "/test_docker/input.nii.gz", "/test_docker/init.pkl", "/test_docker/seg.nii.gz"]
CMD ["python", "vessel_segmentation.py", "coroT2Cube.nii.gz", "init.pkl", "artery", "/test_docker/seg.nii.gz"]
#CMD ["python", "test_docker.py", "/test_docker/input.nii.gz", "/test_docker/init.pkl", "/test_docker/seg.nii.gz"]
