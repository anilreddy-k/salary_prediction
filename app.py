from flask import Flask,render_template,request
app = Flask(__name__)
# model 
import pickle
with open('reg_model.pkl','rb') as f:
    model = pickle.load(f)

#load scalar
with open('scaler.pkl','rb') as f:
    scalar = pickle.load(f)

@app.route('/')
def home():
    # return render_template(<h1>'hello'</h1>)
    return render_template('home.html')
@app.route('/response',methods=['POST'])
def response():
    if request.method == 'POST':
        experience = request.form['experience']
        processed_experience = scalar.transform([[experience]])
        result = model.predict(processed_experience)
        result = round(result[0],2)
        send_result = [experience,result]
        return render_template('home.html',results = send_result)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)
