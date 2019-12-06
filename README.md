#  Computer Vision Coursework README - wtxd25

## How to run

Once you have the zip file containing all of the source code and assets, you run the source code by executing the command "python3 main.py". This uses YOLOv3 which is currently running on the CPU as my computer does not seem to support OpenCL which I did not have the time to get fixed. Therefore, if your computer does have access to OpenCL and you want to improve the performance of the code then navigate to line 29 of "recognition.py" and swap out this line for line 28. Once you have done this you can run main.py as above and the performance of the code should be improved. Can switch to sparse stereo ranging using the boolean on line 13 of "main.py". I have also attempted to add a heuristic even though it does not work amazingly well. This can be enabled on line 26 of "main.py". If you would like to see the effects of disabling the WLS filter on the ranging then you can do so by changing the boolean at line 35 of "main.py".


## Explaining the directory

In the root directory of my project we have all the source code that makes the coursework run, these are all separated into modules to ease maintenance and readability. We also have the yolov3.cfg, the pretrained weights from yolo, and the list of names for object recognition.
The video directory contains the video required for submission. The report folder contains the report PDF files for submission. The images directory contains the resulting images produced from the program running on the dataset provided. The assets directory contains an assortment of training files from when I attempted to train my own model (in the report), the weights produced from that, and the TTBB dataset required.
