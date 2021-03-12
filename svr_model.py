
from flask import Flask, render_template

# Khởi tạo Flask
app = Flask(__name__)

# Hàm xử lý request
@app.route("/", methods=['GET'])
def home_page():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
