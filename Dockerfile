FROM jupyter/datascience-notebook:latest

RUN pip install pyinterval

# TO BUILD:
# sudo docker build -t myjupyter .
#
# TO RUN:
# sudo docker run -it --rm -p 8888:8888 myjupyter
