from flask import Flask, render_template, request, Response #flask 불러오기
import time #타임 모듈 불러오기
from lcd import drivers #lcd모듈 불러오기
import os   #멀티프로세싱
from multiprocessing import Process, Value #멀티프로세싱
import RPi.GPIO as GPIO     #피에조부저를 사용하기 위해서 불러오는 gpio모듈
import webbrowser #url 열때 삳용
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from ctypes import c_bool

#얼굴이 감지되었는지 확인
isDetected = Value(c_bool,False)

#led를 키는 함수
def makeRedLedBright():
    PIN_LED = 22
    GPIO.setmode(GPIO.BCM)  #핀번호방식 설정 
    GPIO.setup(PIN_LED, GPIO.OUT)   #PIN모드 설정
    GPIO.output(PIN_LED,GPIO.HIGH)
    time.sleep(3)
    GPIO.output(PIN_LED,GPIO.LOW)
#초록색 led를 키는 함수
def makeGreenLedBright():
    PIN_LED = 25
    GPIO.setmode(GPIO.BCM)  #핀번호방식 설정 
    GPIO.setup(PIN_LED, GPIO.OUT)   #PIN모드 설정
    GPIO.output(PIN_LED,GPIO.HIGH)
#초록색 led를 끄는 함수
def turnOffGreenLed():
    PIN_LED = 25
    GPIO.setmode(GPIO.BCM)  #핀번호방식 설정 
    GPIO.setup(PIN_LED, GPIO.OUT)   #PIN모드 설정
    GPIO.output(PIN_LED,GPIO.LOW)
#경고음을 울리는 함수
def makeAlertSound():
    BUZZER_PIN = 6
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    pwm = GPIO.PWM(BUZZER_PIN, 262)
    pwm.start(50)
    time.sleep(2)
    pwm.ChangeDutyCycle(0)
    pwm.stop()
    GPIO.cleanup()

#lcd에 글자를 출력해주는 함수
def writingToLcdDisplay():
    try:
        i = 0
        while i<6:
            if(isDetected.value == True):
                print("your face is detected!")
                break
            sec = 5
            while sec:  #sec가 0될 때 까지 계속 반복
                display = drivers.Lcd()
                msformat = '00:{:02d}'.format(sec)  #초를 00:nn형식으로 msformat이라는 변수에 저장
                print(msformat) #msformat을 프린트
                display.lcd_display_string(f"your chance: {5-i}/6",1)
                display.lcd_display_string(f"{msformat}", 2)    #lcd모듈에 msformat에 저장된 값 띄우기
                time.sleep(1)   #1초기다리기
                sec -= 1    #현재 초에서 1초를 빼기
            print(f"your chance: {5-i}/6")
            print(isDetected.value)
            i+=1
        if(isDetected.value == False):
            makeAlertSound()
            makeRedLedBright()
    finally:
        print("Cleaning up")

#창을 띄워 얼굴을 인식하는 함수
def faceDetect():
    data_path = 'faces/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

    Training_Data, Labels = [], []

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(Training_Data), np.asarray(Labels))

    print("Model Training Complete!!!!!")

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        if faces is():
            return img,[]

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

        return img,roi

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
            print(f"result:{result}")
            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                display_string = str(confidence)+'% Confidence it is user'
            cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

            if confidence > 75:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
                isDetected.value = True
                print("lll")
                webbrowser.open("http://localhost:5000/unlockpage")
                break
            else:
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)

        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)

        if cv2.waitKey(1)==13:
            break

    cap.release()
    cv2.destroyAllWindows()

#######################################################메인 코드###############################################################

app = Flask(__name__)

#메인페이지
@app.route('/')
def index_home():
    return render_template('index.html')

#잠금해제를 하는 화면으로 이동하는 한다.
@app.route('/lockpage', methods=['GET'])
def index_test():   
    if request.method == 'GET':
        print("GET!")
        Process(target=faceDetect, args=()).start()
        Process(target=writingToLcdDisplay, args=()).start()
        return "<b>please show your face</b>"
    else:
        return "<p>System error..</p>"

#led를 켠다
@app.route('/ledon',methods=['GET'])
def led_on():
    if request.method == 'GET' and isDetected.value == True:
        makeGreenLedBright()
        return render_template('gather.html')
    else:
        return "<p>your face is not detected yet</p>"


#led를 끈다
@app.route('/ledoff',methods=['GET'])
def led_off():
    if request.method == 'GET' and isDetected.value == True:
        turnOffGreenLed()
        return render_template('gather.html')
    else:
        return "<p>your face is not detected yet</p>"

#스마트폰 화면으로 넘어간다.
@app.route('/unlockpage',methods=['GET'])
def gotounlockpage():
    if request.method == 'GET' and isDetected.value == True:
        return render_template('gather.html')
    else:
        return "<p>your face is not detected yet</p>"

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)