FROM bvlc/caffe:cpu

WORKDIR .

ADD train.py .
ADD requirements.txt .
ADD files files
ADD functions functions

RUN pip install -r requirements.txt

CMD ["python", "train.py", ">", "/log/log_train.txt", "2>&1"]
