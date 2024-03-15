FROM continuumio/miniconda3:main

# copy environment.yml to the image
COPY environment.yml /tmp/env.yml


# create a new environment
RUN conda env create -f /tmp/env.yml  
