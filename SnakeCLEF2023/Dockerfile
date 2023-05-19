FROM nvcr.io/nvidia/pytorch:22.12-py3
# FROM python:3.8.10-slim
# alternatively use python image as a base image, if PyTorch with GPU drivers is not needed

COPY requirements.txt /script/requirements.txt
COPY predict.py /script/predict.py
COPY evaluate.py /script/evaluate.py
COPY start.sh /script/start.sh
COPY ./SnakeCLEF2023-ViT_base_patch16_224-100E.pth /script/SnakeCLEF2023-ViT_base_patch16_224-100E.pth

# install python dependencies
ENV SCRIPT_DIR='/script'
WORKDIR $SCRIPT_DIR
RUN pip install --no-cache-dir --upgrade pip build && \
    pip install --no-cache-dir --compile -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

# run script
CMD bash start.sh
