FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install dependency
COPY app/requirements.txt .
RUN pip install -r requirements.txt

#Copy code inference to standart SageMaker path
COPY app/inference.py /opt/ml/code/inference.py

# Tell SageMaker which file to run
ENV SAGEMAKER_PROGRAM inference.py