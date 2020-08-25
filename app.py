from flask import Flask, request, jsonify, render_template,send_file,make_response
from flask_caching import Cache

#online
import sys
#if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import subprocess
import cv2

import base64
import run_one_im
from pathlib import Path
import os
import io
from PIL import Image

app = Flask(__name__)
config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple", # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 0
}
app.config.from_mapping(config)
cache = Cache(app)


# prevent cached responses
if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

cnt_shot = 0
query_path = "static/img/data/query.jpg"
tmp_query_path = "static/img/data/query_set"
tmp_query_cnt = 1

class CameraHandler(object):
    def __init__(self, rgb_provider="./ros_camera/rgb_provider.sh"):
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

h = CameraHandler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/support_image',methods=['POST'])
def support_image():
    global query_path
    global cnt_shot

    if cnt_shot < 10:
        cnt_shot += 1
    pre_shot  = cnt_shot -1

    data = request.form.to_dict()

    ### save support image from web
    support_im_b64 = data['support_im'].split(',')[1]
    support_im_bytes = base64.b64decode(support_im_b64)
    support_im = Image.open(io.BytesIO(support_im_bytes))
    support_root_dir = 'static/img/data/support'
    class_dir = 'test'
    support_im_path = os.path.join(support_root_dir, class_dir,str(cnt_shot) + '.jpg')
    support_im.save(support_im_path)

    ###fake support image
    class_dir = 'mug2'

    ###fake query image
    #query_path = 'static/img/data/query_demo_5.jpg'


    ### set ouput folder
    output_path_folder = 'static/img/data/result'



    cnt_support_im_paths = []
    for i in range(1,cnt_shot+1):
        cnt_support_im_paths.append(os.path.join(support_root_dir, class_dir, str(i)+'.jpg'))

    pre_support_im_paths = cnt_support_im_paths[:-1]


    # current_support_path_list = []
    # #For rendering html
    # for path  in support_im_paths:
    #     current_support_path_list.append(str(path))
    # n_shot = len(current_support_path_list)


    run_one_im.run_model(cnt_support_im_paths,query_path,cnt_shot,output_path_folder)
    cnt_result_im_path = os.path.join(output_path_folder, 'result' + str(cnt_shot) + '.jpg')
    if pre_shot != 0:
        pre_result_im_path = os.path.join(output_path_folder, 'result' + str(pre_shot) + '.jpg')
    else:
        pre_result_im_path = ''
    # (flag, im_encode) = cv2.imencode(".jpg", result_im)
    # im_bytes = im_encode.tobytes()
    # im_b64 = base64.b64encode(im_bytes)
    #response = {'n_shot':n_shot ,'im_b64' :im_b64}
    #return jsonify(response)
    response = {
        'cnt_shot':cnt_shot,
        'cnt_result_im_path':cnt_result_im_path,
        'cnt_support_im_paths': cnt_support_im_paths,
        'pre_shot': pre_shot,
        'pre_result_im_path': pre_result_im_path,
        'pre_support_im_paths': pre_support_im_paths,
    }
    return jsonify(response)
    # return render_template("index.html",  current_img_path = cnt_result_im_path, current_n_shot = cnt_shot, current_support_path_list =  cnt_support_im_paths ,
    #                        previous_img_path= pre_result_im_path, previous_n_shot= pre_shot,previous_support_path_list= pre_support_im_paths)

@app.route('/query_image',methods=['POST','GET'])
def query_image():
    global cnt_shot
    global h
    global query_path

   
    cnt_shot = 0

    #query_im_b64 = h.get_image()
    #query_im_bytes = base64.b64decode(query_im_b64)
    #query_im = Image.open(io.BytesIO(query_im_bytes))
    #query_im.save(query_path)
    #return query_im_b64

     #for demo
    query_im = cv2.imdecode(np.fromstring(base64.b64decode(h.get_image()), dtype=np.uint8), cv2.IMREAD_COLOR)[...,:3]
    query_im_crop = query_im[45:650,390:990]
    query_im_rotate = rotate(query_im_crop,-10)
    query_im_crop2 = query_im_rotate[30:-60,50:500]
    cv2.imwrite(query_path,query_im_crop2)
    
    #tmp query set
    # global tmp_query_cnt
    # cv2.imwrite(os.path.join(tmp_query_path,str(tmp_query_cnt) + ".jpg"),query_im_crop2)
    # tmp_query_cnt += 1

    (flag, query_im_encode) = cv2.imencode(".jpg", query_im_crop2)
    query_im_bytes = query_im_encode.tobytes()
    query_im_b64 = base64.b64encode(query_im_bytes)

    return query_im_b64
    

    #fake query image
    # im = cv2.imread('static/img/data/query_demo_5.jpg')
    # (flag, im_encode) = cv2.imencode(".jpg",im)
    # im_bytes = im_encode.tobytes()
    # im_b64 = base64.b64encode(im_bytes)
    # return im_b64

    
    
    # global im
    # im = cv2.imread('datasets/query/query_horse.jpg')
    # (flag, im_encode) = cv2.imencode(".jpg",im)
    # im_bytes = im_encode.tobytes()
    # im_b64 = base64.b64encode(im_bytes)
    # return im_b64
    #return jsonify({'query_path':query_path})
    # return render_template("index.html", current_img_path = query_path)

@app.route('/take_a_shot',methods=['POST','GET'])
def take_a_shot():
    global h

    #im_b64 = h.get_image()


    ###fake support image
    global cnt_shot

    #tmp
    #global tmp_query_cnt
    #tmp_query_cnt -= 1

    im = cv2.imread('static/img/data/support/mug2/'+str(cnt_shot+1)+'.jpg')
    (flag, im_encode) = cv2.imencode(".jpg", im)
    im_bytes = im_encode.tobytes()
    im_b64 = base64.b64encode(im_bytes)


    #im = cv2.imdecode(np.fromstring(base64.b64decode(h.get_image()), dtype=np.uint8), cv2.IMREAD_COLOR)[...,:3]

    #cam = cv2.VideoCapture(0)
    #ret, im = cam.read()
    #(flag, im_encode) = cv2.imencode(".jpg", im)
    #im_bytes = im_encode.tobytes()
    #im_b64 = base64.b64encode(im_bytes)
    return im_b64

    # global im
    # im = cv2.imread('datasets/query/query_horse.jpg')
    # return jsonify({'query_path':query_path})
    # return render_template("index.html", current_img_path = query_path)

def rotate(image, angle, center=None, scale=1.0):

    (h, w) = image.shape[:2]
   
    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

if __name__ == "__main__":
    app.run(debug=True)
