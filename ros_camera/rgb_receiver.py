import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import json
import base64
import subprocess

class CameraHandler(object):
    def __init__(self, rgb_provider="./rgb_provider.sh"):
        self.expect = 'b64_img: '
        self.sub_process = subprocess.Popen((rgb_provider,), stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    def get_image(self):
        command = b"get_image\n"
        self.sub_process.stdin.write(command)
        self.sub_process.stdin.flush()
        while True:
            response_b64 = self.sub_process.stdout.readline().decode("utf-8").strip()
            sys.stdout.flush()
            if response_b64.startswith(self.expect):
                return response_b64[len(self.expect):]
    def __del__(self):
        self.sub_process.kill()

if __name__ == '__main__':
    h = CameraHandler()
    while True:
        img = cv2.imdecode(np.fromstring(base64.b64decode(h.get_image()), dtype=np.uint8), cv2.IMREAD_COLOR)[...,:3]
        cv2.imshow('show img', img)
        if cv2.waitKey(-1) == ord('q'):
            break
