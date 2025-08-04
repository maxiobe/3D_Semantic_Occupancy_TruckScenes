# Use a modern, official ROS Melodic image as the base
FROM osrf/ros:melodic-desktop-full-bionic

# Install Python 3, pip, and essential build tools
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Use pip to install numpy
RUN pip3 install numpy PyYAML rospkg pycryptodomex python-gnupg

# Set the default working directory
WORKDIR /scripts