#FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.1.0-gpu-py310

# Install dependency
COPY app/requirements.txt .
RUN pip install -r requirements.txt

#Copy code inference to standart SageMaker path
COPY app/inference.py /opt/ml/code/inference.py

# Tell SageMaker which file to run
ENV SAGEMAKER_PROGRAM = inference.py

# Optional local model path override
ENV MODEL_DIR = /opt/ml/model
