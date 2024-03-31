# Environment Setup Guide

## Table of Contents
- [Environment Setup Guide](#environment-setup-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Cloning the Repository](#cloning-the-repository)
  - [Running the Docker Container](#running-the-docker-container)
  - [Executing Commands Inside the Container](#executing-commands-inside-the-container)
  - [Check the status of the container](#check-the-status-of-the-container)
  - [Starting and Stopping the Container](#starting-and-stopping-the-container)
  - [Conclusion](#conclusion)


## Introduction
This guide is intended for individuals new to Docker and explains how to interact with the "DSC-180B-Hardware-Acceleration" Docker image. This image includes a setup with Ubuntu 22.04, Python 3.9 (with Conda), PyTorch, and other tools.

**Current Limitation**: Currently the docker image does not support GPU acceleration. It will be added in the future.

## Prerequisites
- Docker installed on your machine. For installation instructions, visit [Docker's official website](https://www.docker.com/get-started).

## Cloning the Repository
1. **Clone the Repository**: Clone the repository using:
   ```bash
   git clone https://github.com/louiecai/DSC-180B-Hardware-Acceleration.git
    ```
2. **Navigate to the Repository**: Navigate to the repository using:
    ```bash
    cd DSC-180B-Hardware-Acceleration
    ```

## Building the Docker Image
1. **Obtain the Dockerfile**: Make sure you have the Dockerfile for "DSC-180B-Hardware-Acceleration".
2. **Navigate to the Dockerfile Directory**: Open a terminal and go to the directory containing the Dockerfile.
3. **Build the Image**: Run the command:
   ```bash
   docker build -t dsc180b-hardware-acceleration .
   ```
   This will build the Docker image and tag it as `dsc180b-hardware-acceleration`.


## Running the Docker Container
**Run the Container with Volume Mounting**: To create the container and mount a volume from your host to the container, use:
   ```bash
   docker run -it -v $(pwd):/root/home/DSC-180B-Hardware-Acceleration dsc180b-hardware-acceleration /bin/zsh
   ```
   Here, `-it` enables interactive mode, `-v $(pwd):/root/home/DSC-180B-Hardware-Acceleration` mounts the current directory (`$(pwd)`) to `/root/home/DSC-180B-Hardware-Acceleration` in the container, and `/bin/zsh` starts a Zsh shell.

## Executing Commands Inside the Container
- To execute additional commands inside the container, you can attach to it using:
  ```bash
  docker exec -it [container_name_or_id] /bin/zsh
  ```
  Replace `[container_name_or_id]` with the actual name or ID of your container.

## Check the status of the container
- **Check the status of the container**: To check the status of your container, use:
  ```bash
  docker ps -a
  ```
  This will list all containers on your machine, including stopped containers.

## Starting and Stopping the Container
- **Start the Container**: If your container is stopped, start it with:
  ```bash
  docker start [container_name_or_id]
  ```
- **Stop the Container**: To stop a running container, use:
  ```bash
  docker stop [container_name_or_id]
  ```
## Conclusion
This README provides the basics for building, running, interacting with, starting, and stopping the "DSC-180B-Hardware-Acceleration" Docker container. For more comprehensive information, refer to the [Docker documentation](https://docs.docker.com/).