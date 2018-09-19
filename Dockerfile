FROM bvlc/caffe:cpu

ADD vessel_segmentation.py .

CMD ["python", "vessel_segmentation.py"]
