#!/usr/bin/env /home/test/pyenv_pyrobot/bin/python
import sys
import os
import numpy as np

import io
import base64
import json
import subprocess

pyrobot_path = os.path.expanduser("~/pyrobot/src")
sys.path.append(pyrobot_path)
from pyrobot.core import Robot
from pyrobot import util
import rospy
import cv2
import numpy as np

if __name__ == '__main__':
    bot = Robot("ur3")
    while True:
        try:
            line = raw_input()
        except EOFError:
            break
        if not line.startswith("get_image"):
            continue
        im = bot.camera.get_rgb_depth()[0][...,::-1]
        (flag, im_encode) = cv2.imencode(".jpg", im)
        im_bytes = im_encode.tobytes()
        im_b64 = base64.b64encode(im_bytes)
        print('b64_img: ' + im_b64)
        sys.stdout.flush()

