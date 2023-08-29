FROM openvino/ubuntu20_dev:2022.1.0
COPY requirements.txt  requirements.txt 
RUN pip3 install -r requirements.txt
RUN pip3 install ipykernel
