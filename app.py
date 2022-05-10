from flask import Flask, render_template, jsonify, request
# render_template: template 파일을 읽어오기 위한 것

app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

## API 역할을 하는 부분
@app.route('/review', methods=['POST'])
def write_review():
    sample_receive = request.form['sample_give']
    print(sample_receive)
    return jsonify({'msg': '이 요청은 POST!'})
@app.route('/review', methods=['GET'])
def read_reviews():
    sample_receive = request.args.get('sample_give')
    print(sample_receive)
    return jsonify({'msg': '이 요청은 GET!'})


# render_template: template 파일 중 index.html을 불러 옴 
if __name__ == '__main__':
   app.run('0.0.0.0',port=5000,debug=True)