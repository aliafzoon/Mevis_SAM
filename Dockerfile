FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

RUN apt update && apt upgrade -y
RUN apt install wget build-essential libgl1-mesa-glx libglib2.0-0 -y
RUN pip install --upgrade pip

RUN pip install typer numpy matplotlib pandas opencv-python torchio
RUN pip install jupyter
# RUN pip install --upgrade --force-reinstall torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/test/cu121
RUN pip install monai-weekly timm pynrrd einops slicerio scikit-image tqdm SimpleITK
# RUN jupyter server --generate-config
# RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py
COPY ./jupyter_notebook_config.py /root/.jupyter/
# EXPOSE 8888
WORKDIR /notebooks
# Start Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]


