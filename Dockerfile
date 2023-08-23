FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
COPY requirements.txt  requirements.txt 
RUN pip3 install -r requirements.txt
RUN pip3 install ipykernel
