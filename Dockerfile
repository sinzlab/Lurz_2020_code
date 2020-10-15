FROM sinzlab/pytorch:v3.8-torch1.5.0-cuda10.2-dj0.12.4
RUN pip install --upgrade pip

RUN pip install 'neuralpredictors~=0.0.1'
RUN pip install --pre nnfabrik

ADD . /src/lurz2020
RUN pip install -e /src/lurz2020

WORKDIR /notebooks

COPY ./jupyter/jupyter_notebook_config.py /root/.jupyter/
