# UOW-Agriculture-Threats-Detection
Detecting weeds and plant disease using state of the art computer vision models

## How to Run with Docker

### 1 - Downloading model weight

Download best-run2.onnx weight file from [here](https://drive.google.com/file/d/1sZ89FV65LeU4ZpP5f0Lz3r5yPaSc3yQi/view?usp=sharing) or find it included in submissables, and put in the **weights folder**. You can also download a second weights version [here](https://drive.google.com/file/d/11pjiQL2H5mKhTC-fkzECUl2cYeg65aJQ/view?usp=sharing) and rename it best-run2.onnx or go to utils/utils and change the get_weights_path() function


### 2 - Docker Requirements

Before you build Docker Image, make sure you have enough space in your system because the docker image will take 5GB of your storage. 
Install docker desktop and open it. Ensure an engine is running.

- Storage - 8.75GB
- RAM - 4GB/8GB

The docker desktop version I tested is 4.0.1 (68347). However, you should be able to build image with latest docker destop too.

### 3 - Builing Docker Image

- go to the root dir (**UOW-AGRICULTURE-THREATS-DETECTION**)
- open docker desktop 
- run below command in your terminal to build docker image

```bash
docker compose build
```

After succesful build, you should be seeing something similar as shown below.

<img src="assets/sample-1.png" width="800" height="400"><br>

### 4 - Running Docker Container

To run your container, run below command

```bash
docker compose up
```

Open one of the IP addresses as shown in your terminal or type localhost:8888 and it will direct you to Flask Web Application.

<img src="assets/sample-2.png" width="950" height="150"><br>
