FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get -y install python3 python3-pip python3-openslide git python3-setuptools \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy==1.20.1 scikit-learn==0.24.1 scipy==1.6.1 Pillow==8.1.1 pydicom==0.9.9 imageio==2.9.0 \
scikit-image==0.18.1 boto3==1.17.18 tensorflow-gpu==2.4.1 matplotlib==3.3.4

WORKDIR /home
RUN git clone https://github.com/ncoudray/DeepPATH.git
COPY src/script.py .
COPY tcga-svs-labels tcga-svs-labels

# Make sure python doesn't buffer stdout so we get logs ASAP.
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3", "script.py"]