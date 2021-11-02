from flask import Flask, render_template, request, redirect, url_for
import sys
import model
import clova
import requests

application = Flask(__name__)

@application.route("/" )
def hello():
    #location = request.args.get('location') #input에서 required name
    #cleaness = request.args.get('clean') #checkbox에서 value
    #built_in = request.args.get('built') #textarea에서 name
    #print(location,cleaness, built_in)
    #stt_text=''
    #summ_text=''
    
    #stt_text, summ_text=model.model_start()
    return render_template("test.html")

@application.route("/upload_done", methods=["POST"])
def upload_done():
    upload_files = request.files['file']
    #upload_files.save('static/img/{}.jpeg'.format(1))
    upload_files.save('static/sound/{}.m4a'.format(1))
    #stt_text, summ_text=model.model_start()
    
    return redirect(url_for("apply_"))

    # hello라는 함수로 redirct -> home 화면
    
@application.route("/apply")
def apply_():
    stt_text, summ_text=model.model_start()
    return render_template("apply.html", stt_text=stt_text, summ_text=summ_text)
    

if __name__ == "__main__":
    application.run(host='0.0.0.0')
    #port=int(sys.argv[1])

