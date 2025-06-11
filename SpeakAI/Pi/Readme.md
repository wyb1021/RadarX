https://www.raspberrypi.com/documentation/accessories/ai-camera.html#model-deployment
위 링크로 접속해서 ai 카메라 예제들 실행해보기

rpicam-vid -t 10s -o output.264 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json --width 1920 --height 1080 --framerate 30 
위 코드도 하나의 예제