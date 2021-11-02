import sys
import requests
import re

def clova_start():
    client_id = "lvqtx1nog1"
    client_secret = "qkcpcSnIzBaOOQLdKwRY4uqUDM3LOECSzxOZkZ1k"
    lang = "Kor" # 언어 코드 ( Kor, Jpn, Eng, Chn )
    url = "https://naveropenapi.apigw.ntruss.com/recog/v1/stt?lang=" + lang
    data = open('static/sound/1.m4a', 'rb')
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
        "Content-Type": "application/octet-stream"
    }
    response = requests.post(url,  data=data, headers=headers)
    rescode = response.status_code
    if(rescode == 200):
        clova_result=response.text
        #print (response.text)
    else:
        print("Error : " + response.text)
    clova_result=re.sub('[^ .A-Za-z0-9가-힣+]','',re.sub('{"text":"', '',clova_result))
    return clova_result

if __name__ =="__main__":
    clova_start()