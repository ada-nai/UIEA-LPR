# UIEA-LPR
\
License Plate recognition from images of Chinese vehicles using [pre-trained models](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models) provided under the [Intel® Distribution of OpenVINO™ Toolkit](https://software.intel.com/en-us/openvino-toolkit)  
\

![Input - License plate](https://raw.githubusercontent.com/ada-nai/UIEA-LPR/master/test5.jpg?token=AMH2MUUUT2PIGZPLLGGBYQC6I6VPM)
![Output - Recognized License Plate Number](https://raw.githubusercontent.com/ada-nai/UIEA-LPR/master/op_test5.png?token=AMH2MUVPF5MYOGIQF3XAK6S6I6VTS)

Following pre-trained models were used for the application:

  1) [vehicle-license-plate-detection-barrier-0106](https://docs.openvinotoolkit.org/2019_R1/_vehicle_license_plate_detection_barrier_0106_description_vehicle_license_plate_detection_barrier_0106.html)
  2) [license-plate-recognition-barrier-0001 ](https://docs.openvinotoolkit.org/2019_R1/_license_plate_recognition_barrier_0001_description_license_plate_recognition_barrier_0001.html)

```
vpd.py    -> vehicle/plate detection app
vinfer.py -> inference helper file for vpd.py
          
lpr.py    -> license plate recognition app
linfer.py -> inference helper file for lpr.py
```

~~`car.bmp` was used in the pre-trained model test
\
other images are kept for future reference~~
```
car_image -> vpd.py -> output blob of detected plate -> lpr.py -> output of plate number
```
