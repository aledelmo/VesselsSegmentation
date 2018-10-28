FROM bvlc/caffe:cpu

WORKDIR .

ADD vessel_segmentation.py .
ADD requirements.txt .
ADD processing_vs processing_vs
ADD cnn_models cnn_models

ADD test_dataset/docker_infer/reference.nii .
ADD test_dataset/docker_infer/arteries.pkl .
ADD test_dataset/docker_infer/veins.pkl .

RUN pip install -r requirements.txt

CMD ["python", "vessel_segmentation.py", "reference.nii",  "/test_docker/seg-label.nii", "--arteries", "arteries.pkl", "--veins", "veins.pkl"]
