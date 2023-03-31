#　魚体が銀色系の場合に使う処理
from flask import Flask, jsonify, request
import base64
import cv2
import tempfile
import os
import numpy as np
import requests
import urllib.parse

app = Flask(__name__)

SENSOR_SIZE = {
    "iPhone 8": (3.99, 2.95),
    "iPhone 8 Plus": (5.7, 4.28),
    "iPhone X": (6.0, 4.5),
    "iPhone XR": (6.86, 5.14),
    "iPhone XS": (5.68, 4.26),
    "iPhone XS Max": (6.24, 4.68),
    "iPhone 11": (5.68, 4.26),
    "iPhone 11 Pro": (5.67, 4.26),
    "iPhone 11 Pro Max": (6.22, 4.68),
    "iPhone SE (2nd generation)": (4.89, 3.67),
    "iPhone 12": (5.78, 4.34),
    "iPhone 12 Mini": (5.68, 4.26),
    "iPhone 12 Pro": (5.78, 4.36),
    "iPhone 12 Pro Max": (6.33, 4.75),
    "iPhone 13": (5.78, 4.34),
    "iPhone 13 Mini": (5.68, 4.26),
    "iPhone 13 Pro": (5.78, 4.36),
    "iPhone 13 Pro Max": (6.33, 4.75),
}


def get_pixel_size(generation, exif):
    # 必要な値を取得
    sensor_width, sensor_height = SENSOR_SIZE[generation]
    focal_length = exif["FocalLength"]
    sensor_resolution = exif["PixelXDimension"] / sensor_width
    
    # 1pxあたりの実際の長さを計算
    pixel_size = sensor_width * focal_length / (sensor_resolution * exif["PixelXDimension"])
    
    return pixel_size


@app.route("/", methods=["POST"]) # FlaskAPIのエンドポイント内で以下のように使用する
def process_image():
    
    color = request.json["color"]
    generation = request.json["generation"]
    exif = request.json["exif"]
    image = request.json["image"]
    response = requests.get(image)
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(response.content)
    local_path = temp_file.name
    temp_file.close()
    print(color, generation, exif)

    parsed_uri = urllib.parse.urlparse(image)
    filename = os.path.basename(parsed_uri.path)
    print(local_path, filename)

    if color == "white":
        pixel_size = get_pixel_size(generation, exif)
        # 画像の読み込み
        img = cv2.imread(local_path)

        # ガウシアンフィルタによる平滑化
        img_blur = cv2.GaussianBlur(img, (9, 9), 0)

        # 画像のHSV変換
        hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

        #銀色やグレーに近い色相、彩度、明度の範囲を指定
        lower_color1 = np.array([0, 35, 0])
        upper_color1 = np.array([180, 110, 75])
        lower_color2 = np.array([30, 20, 0]) # 変更
        upper_color2 = np.array([220, 110, 75])

        #指定した範囲の色に含まれるピクセルを抽出
        mask1 = cv2.inRange(hsv, lower_color1, upper_color1)
        mask2 = cv2.inRange(hsv, lower_color2, upper_color2)
        mask = cv2.bitwise_or(mask1, mask2)

        #クロージングによるノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        #輪郭の検出
        contours, _ = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #面積が小さい輪郭を除外
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]

        #最大の輪郭を取得
        max_contour = max(contours, key=cv2.contourArea)

        #輪郭の描画
        img_contours = cv2.drawContours(img.copy(), [max_contour], -1, (0, 0, 255), 2)

        #輪郭の近似
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        #最も離れた2点間の座標を取得
        max_dist = 0
        for i in range(len(approx)):
            for j in range(i+1, len(approx)):
                temp_dist = np.linalg.norm(approx[i] - approx[j])
                if temp_dist > max_dist:
                    max_dist = temp_dist
                    max_dist_points = [tuple(approx[i][0]), tuple(approx[j][0])]

        #ピクセル単位での距離計算
        pixel_distance = max_dist
        real_lengths = pixel_distance * pixel_size


        return jsonify({"計測結果":real_lengths})
    else:
        # その他の色に対する処理を実行
        return jsonify({"error": "Invalid color"})

if __name__ == '__main__':
    app.run()