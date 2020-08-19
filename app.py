from flask import Flask, request, jsonify, render_template,send_file,make_response
from flask_caching import Cache

import cv2
import base64
import run_one_im
from pathlib import Path
import os

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
query_path = 'static/img/data/query01.jpg'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/support_image',methods=['POST','GET'])
def support_image():
    global cnt_shot
    #a =request.form.to_dict()
    #print(a['number'])

    support_root_dir = 'static/img/data/support'
    class_dir = 'horse'

    output_path_folder = 'static/img/data/result'
    #support_im_paths = list(Path(os.path.join(support_root_dir, class_dir)).glob('*.jpg'))
    if cnt_shot < 5:
        cnt_shot += 1
    pre_shot  = cnt_shot -1

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
    cnt_shot = 0
    # global im
    # im = cv2.imread('datasets/query/query_horse.jpg')
    # (flag, im_encode) = cv2.imencode(".jpg",im)
    # im_bytes = im_encode.tobytes()
    # im_b64 = base64.b64encode(im_bytes)
    # return im_b64
    return jsonify({'query_path':query_path})
    # return render_template("index.html", current_img_path = query_path)


if __name__ == "__main__":
    app.run(debug=True)