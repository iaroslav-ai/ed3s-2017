FROM ubuntu:16.04
MAINTAINER Iaroslav Shcherbatyi

ADD source /root/source
RUN /bin/bash /root/source/dependencies.sh

CMD ["/bin/bash", "python3 -m jupyterhub"]
