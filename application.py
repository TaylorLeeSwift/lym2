from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import time
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)

# 读取城市数据
city_data = pd.read_csv('us-cities.csv')

# 每页城市数
cities_per_page = 50


def haversine(lat1, lon1, lat2, lon2):
    # 将经纬度从度数转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine公式计算距离
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius_of_earth = 6371  # 地球半径（单位：公里）

    # 计算距离（单位：千米）
    distance = radius_of_earth * c

    return distance

def calculate_distances(target_city, target_state):
    # 查询城市距离并计算响应时间
    start_time = time.time()

    # 初始化结果列表
    city_distances = []

    # 获取目标城市的经纬度
    target_city_data = city_data[(city_data['city'] == target_city) & (city_data['state'] == target_state)]
    if not target_city_data.empty:
        target_lat = float(target_city_data['lat'])
        target_lng = float(target_city_data['lng'])

        # 计算目标城市与其他城市的距离
        for _, city in city_data.iterrows():
            lat = float(city['lat'])
            lng = float(city['lng'])
            distance = haversine(target_lat, target_lng, lat, lng)
            city_distances.append(distance)

    response_time = int((time.time() - start_time) * 1000)  # 计算响应时间（毫秒）
    return city_distances, response_time


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city_name = request.form['city']
        state_name = request.form['state']

        # 计算城市距离
        distances, response_time = calculate_distances(city_name, state_name)

        # 将数据转换为JSON格式
        data = json.dumps({'distances': distances, 'response_time': response_time})
        return render_template('index.html', data=data)

    return render_template('index.html', data=None)


if __name__ == '__main__':
    app.run(debug=True)