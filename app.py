# 参考URL https://engineer-lifestyle-blog.com/code/python/flask-tutorial-web-app-with-database/
# 必要なモジュールのインポート
#import torch
#from animal import transform, Net # animal.py から前処理とネットワークの定義を読み込み
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64
#from flask_sqlalchemy import SQLAlchemy

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import pytorch_lightning as pl
#import pytesseract
import re
import _thread
import sqlite3
import numpy as np
import os

from torchvision.models import resnet18

#OCR実行関数
#cv2.resize()にて拡大(Bicubic補間にて4倍に拡大)後、OCR実行するサンプル
#def img_to_text(img):
#    resized = cv2.resize(img, (img.shape[1]*8, img.shape[0]*8),interpolation=cv2.INTER_CUBIC)
#    result = pytesseract.image_to_string(img, lang="jpn", config="--psm 6")
#    result = result.replace('\n', '')
#    result = result.replace('\x0c', '')
#    return result

# Resnet18に合わせた前処理を追加
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Net(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h

net = Net().cpu().eval()
# 重みの読み込み
net.load_state_dict(torch.load('id_no.pt', map_location=torch.device('cpu')))
def mnist(img):
    img = transform(img)
    img =img.unsqueeze(0) # 1次元増やす
    y = net(img)
    y = F.softmax(y, dim=1)
    y = torch.argmax(y)
    y = y.item()
    return(y)

# オーダーリストを作る
def my_order(file_full_path):
    img = cv2.imread(file_full_path)
    #template画像("個数")の座標抽出
    # テンプレート画像を読み込み、グレースケールに変換
    template = cv2.imread('template.png', 0)
    # 入力画像が3次元以上の場合は2次元に変換
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # テンプレートマッチングを実行
    temp_result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # 一致マップから最もマッチした箇所の座標を取得
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_result)

    # 閾値を設定して、マッチング度が高い箇所の位置を取得する
    threshold = 0.75
    locs = np.where(temp_result >= threshold)
    x_coords = locs[1] # テンプレート画像の左上のx座標を取得する

    template_height = template.shape[0]  # テンプレート画像の高さ
    max_y = locs[0].max() + template_height  # 検出された領域の左下の y 座標

    #テンプレート画像の下から対象画像を抽出
    # エッジ抽出 (Canny)
    gray_img = cv2.imread(file_full_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(gray_img, 1, 300, apertureSize=3)
    
    # 膨張処理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilates = cv2.dilate(edges, kernel)
    # 輪郭抽出
    #contours, hierarchy = cv2.findContours(dilates, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(dilates, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # オーダーリストを初期化する
    order_list = []
    #-----------------------------------------入れ替え
    #複数の変数を取得するときはZip enumerateはインデックス
    for i, (cnt, hrchy) in enumerate(zip(contours, hierarchy[0])):
        if cv2.contourArea(cnt) < 600:   #調整が必要
            continue  # 面積が小さいものは除く
        if hrchy[3] == -1:
            continue  # ルートノードは除く
        # 輪郭を囲む長方形を計算する。
        rect = cv2.minAreaRect(cnt)  #輪郭に外接する回転した長方形
            #retval: ((中心の x 座標, 中心の y 座標), (長方形の幅, 長方形の高さ), 回転角度) 
        box = cv2.boxPoints(rect)         #長方形4点の座標 　左上、右上、右下、左下
 
        # 配列構造が多重になっているので、１回シンプルな配列にする (point [x y])
        contour = []
        contour = list(map(lambda point: point[0], cnt))
        x_list = [ point[0] for point in contour ] # width
        y_list = [ point[1] for point in contour ] # height
        #return  [min(x_list), max(x_list), min(y_list), max(y_list)]
        left_top_y = min(y_list)
        x = min(x_list)
        y = min(y_list)
        width = template.shape[1]
        height = 28
        #angle = rect[2]

        # x_coords に格納されている座標のプラスマイナス10の範囲内にあるかどうかを判定する
        if any(abs(template_x - x) <= 10 for template_x in x_coords) and (left_top_y > max_y):
            img_cut = img[int(y):int(y+height), int(x):int(x+width)]
            if img_cut.shape == (0,0):
                continue
            #cv2.imwrite('imgs/imgcut' + str(i) + '.png', img_cut) 
            #if len(img_cut.shape) == 2:
            #    img_cut = cv2.cvtColor(img_cut, cv2.COLOR_GRAY2BGR)
 
            #青の色相の範囲を指定する
            #色相（Hue）、彩度（Saturation）、明度（Value）を指定
            # BGR色空間からHSV色空間に変換する
            #hsv = cv2.cvtColor(img_cut, cv2.COLOR_BGR2HSV)
        
            #lower_blue = np.array([50, 50, 50])
            #upper_blue = np.array([180, 255, 255])

            # 色の範囲にマスクを作成する
            # img_cutが真っ白の場合はマスクを全て0の画像として作成
            #if np.all(img_cut == 255):
            #    img_mask = np.zeros_like(img_cut, dtype=np.uint8)
            #else:
            #img_mask = cv2.inRange(img_cut, lower_blue, upper_blue)

            #IDの範囲で切り出し
            img_mask = img[int(y):int(y+28), int(x):int(x+template.shape[1]*0.2)]
            if img_mask is not None:                
                blur = cv2.GaussianBlur(img_mask, (3, 3), 1)
                # Cannyエッジ検出を適用する
                edges_cut = cv2.Canny(blur , 200, 500)
                #メニューIDが映っているもののみにする      
                if cv2.countNonZero(edges_cut) < 10:
                    continue            
                     
            #個数の入力があるか
            # マスクを反転して、抽出した部分のみ白にする
            #img_mask_inv = cv2.bitwise_not(img_mask)
            #マスクのチャンネルを増やす
            #img_mask3 = cv2.cvtColor(img_mask , cv2.COLOR_GRAY2BGR)
            #マスクの膨張処理
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            #img_mask4 = cv2.dilate(img_mask3, kernel)
            #青文字部分を白へ
            #img_black = cv2.bitwise_or(img_cut, img_mask4)
        
            img_black = img_cut[0:28, int(template.shape[1]*0.2):]
            # ガウシアンフィルタを適用する
            blur = cv2.GaussianBlur(img_black, (3, 3), 1)
            # Cannyエッジ検出を適用する
            edges_cut = cv2.Canny(blur , 200, 500)  
            if cv2.countNonZero(edges_cut) < 10:
                continue            
            cv2.imwrite('imgs/img_cut.png', img_cut) 
            
        #-----------------------------------------入れ替え         

            keta = 0
            #img_gray = Image.open('imgs/img_cut.png').convert('L')
            img_cut2 = Image.open('imgs/img_cut.png')
            # 画像の幅と高さを取得
            width = template.shape[1]
            height = 28
            # トリミングする幅を計算
            trim_width = width*0.068
            # トリミングする範囲を指定
            x = 1
            y = 3

            # 画像1をトリミング
            img_1 = img_cut2.crop((x, y, x + trim_width, y + height-8))
            img_array = np.array(img_1)
            # 閾値を設定して2値化する
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # ガウシアンフィルタを適用する
            blur = cv2.GaussianBlur(img_bin, (3, 3), 1)
            # Cannyエッジ検出を適用する
            edges_cut = cv2.Canny(blur , 200, 500)  
            # 画像の非ゼロピクセル数をカウントする
            count = cv2.countNonZero(edges_cut)
            #if count > 40:
            keta = 1
            # 画像を28x28にリサイズする
            #img = cv2.resize(img_array, (28, 28))
            # テンソルに変換する
            #img_tensor = img.reshape((1, 1, 28, 28))
            #img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, 28, 28)に変換し、値を0〜1に正規化する
            img1 = mnist(img_1)
                        
            # 画像2をトリミング
            img_2 = img_cut2.crop((x + trim_width, y, x + trim_width*2, y + y + height-8))
            img_array = np.array(img_2)
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            # 閾値を設定して2値化する
            _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # ガウシアンフィルタを適用する
            blur = cv2.GaussianBlur(img_bin, (3, 3), 1)
            # Cannyエッジ検出を適用する
            edges_cut = cv2.Canny(blur , 200, 500)  
            # 画像の非ゼロピクセル数をカウントする
            count = cv2.countNonZero(edges_cut)
            if count > 40:
                keta = 2
                # 画像を28x28にリサイズする
                #img = cv2.resize(img_array, (28, 28))
                # テンソルに変換する
                #img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, 28, 28)に変換し、値を0〜1に正規化する
                img2 = mnist(img_2)

            # 画像3をトリミング
            img_3 = img_cut2.crop((x + trim_width*2, y, x + trim_width*3, y + y + height-8))
            img_array = np.array(img_3)
            # 閾値を設定して2値化する
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            _, img_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # ガウシアンフィルタを適用する
            blur = cv2.GaussianBlur(img_bin, (3, 3), 1)
            # Cannyエッジ検出を適用する
            edges_cut = cv2.Canny(blur , 200, 500)  
            # 画像の非ゼロピクセル数をカウントする
            count = cv2.countNonZero(edges_cut)
            if count > 40:
                keta = 3
                # 画像を28x28にリサイズする
                #img = cv2.resize(img_array, (28, 28))
                # テンソルに変換する
                #img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, 28, 28)に変換し、値を0〜1に正規化する
                img3 = mnist(img_3)

            #id
            if keta == 1:
                id = img1
            elif keta == 2:
                id = img1*10 + img2
            elif keta == 3:
                id = img1*100 + img2*10 + img3
            else:
                id = 0
            
            if id > 120:
                id = 120
            print(id)

            #個数を取得
            # Hough変換を行い、正の字から直線を検出
            # 画像の高さ、幅を取得
            width = template.shape[1]
            height = 28
            # トリミングする幅を計算
            trim_width = int(template.shape[1] / 3)
            # 罫線を削除するために回りを削除
            start_x = trim_width
            start_y = 2
            end_x = width-2
            end_y = height-2
            img_cut3 = cv2.imread('imgs/img_cut.png')
            img_trim_r = img_cut3[start_y:end_y,start_x:end_x]
            #img_trim_r = img_cut2[start_y:end_y,start_x:end_x] 
            #gray = cv2.cvtColor(img_trim_r, cv2.COLOR_BGR2GRAY)
            #reversed_gray = cv2.bitwise_not(gray)
            edges = cv2.Canny(img_trim_r, 200, 500, apertureSize=3)
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=20, minLineLength=15, maxLineGap=3)
            if lines is not None:
                kosu = len(lines)
            else:
                kosu = 0

            #order_listに追加    
            order_list.append([id, kosu])
    return order_list

# Flask のインスタンスを作成
app = Flask(__name__,static_folder='./templates/images')

#　拡張子が適切かどうかをチェック
def allwed_file(filename):
    # アップロードされる拡張子の制限
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def order():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allwed_file(file.filename):

            # アップロード先のディレクトリを指定
            #upload_dir = '../imgs'
            upload_dir = 'imgs'
            # ファイルの保存
            file.save(os.path.join(upload_dir, file.filename))
            # 保存されたファイルのフルパスを取得
            file_full_path = os.path.join(upload_dir, file.filename)
            
            #　画像ファイルに対する処理
            #　画像書き込み用バッファを確保
            #buf = io.BytesIO()
            #image = Image.open(file).convert('RGB')
            #　画像データをバッファに書き込む
            #image.save(buf, 'png')
            #　バイナリデータを base64 でエンコードして utf-8 でデコード
            #base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            #　HTML 側の src  の記述に合わせるために付帯情報付与する
            #base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 入力された画像に対して読み取り実行
            orderlist = my_order(file_full_path)
                        
            #単価DB(register.db)へ接続
            # threading.local() オブジェクトを作成する
            local = _thread._local()
            dbname = 'register.db'
            local.conn = sqlite3.connect(dbname)
            # カーソルオブジェクトを作成
            cur = local.conn.cursor()

            #データを取得
            regi_list=[]
            for order in orderlist:
                
                sql = 'SELECT * FROM regi_article WHERE id = ?'
                params = (order[0],)
                cur.execute(sql, params)
                #sql = 'SELECT * FROM regi_article where id=' & order[0]
                #cur.execute(sql)
                row = cur.fetchone()
                #regi_listにデータ作成
                if row:
                    regi_list.append({
                        "id":row[0],
                        "bunrui1":row[1],
                        "bunrui2":row[2],
                        "name":row[3],
                        "kosu":order[1],
                        "price":row[5],
                        "syokei":int(order[1])*int(row[5])
                    })
                # 合計金額を計算
                total = sum(item['syokei'] for item in regi_list)

            #画面のプルダウンを作るためのデータを取得
            bunrui1_list = ['料理', 'ドリンク']
            bunrui2_list = {}

            for bunrui1 in bunrui1_list:
                sql = 'SELECT distinct bunrui2 FROM regi_article WHERE bunrui1 = ?'
                params = (bunrui1,)
                cur.execute(sql, params)
                rows = cur.fetchall()
                bunrui2_list[bunrui1] = [row[0] for row in rows]
            
            name_list = {}
            for bunrui1 in bunrui1_list:
                for bunrui2 in bunrui2_list[bunrui1]:
                    sql = 'SELECT name, price FROM regi_article WHERE bunrui2 = ? ORDER BY id'
                    params = (bunrui2,)
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    name_list[bunrui2] = [(row[0], row[1]) for row in rows]
            
            #単価DBを閉じる
            local.conn.close()
            return render_template('result.html', regi_list=regi_list, total=total,bunrui1_list=bunrui1_list,bunrui2_list=bunrui2_list,name_list=name_list)         
        
    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')

# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)