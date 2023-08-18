import itertools
import time, socket, sys
from flask import Flask, Response, redirect, request, url_for
from requests import get
import csv
##import pyautogui

app = Flask(__name__)
t = time.time()
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_name = "0.0.0.0"
ip = socket.gethostbyname(host_name)
port = 8080
soc.bind((host_name, port))
my_ip = get("https://api.ipify.org").text
print("'",my_ip,"'으로 접속 가능합니다")
#여기까지가 서버 만들기
name = "ff"
soc.listen(1) #Try to locate using socket
print('클라이언트 연결 대기중')
connection, addr = soc.accept()
print("Received connection from ", addr[0], "(", addr[1], ")\n")
print('Connection Established. Connected From: {}, ({})'.format(addr[0], addr[0]))
#get a connection from client side
client_name = connection.recv(1024)
client_name = client_name.decode()

connection.send(name.encode())
notice = 0
bax = 0
od = 0   
def upti(i,f, floor_direction, check):
    global od

    x = time.localtime()

    with open('걸린시간.csv', 'a', encoding='utf-8', newline='') as csvf:
        od = od+1
        wr = csv.writer(csvf) 
        print([od, x[1], x[2], x[3],x[4], f, floor_direction, round(i,3),check])
        wr.writerow([od, x[1], x[2], x[3],x[4], f, floor_direction, round(i,3),check])

def timechecker(x11, fullcheck):
    global bax, notice, fsa, seb
    if x11[0] == "B":
        x11 = -int(x11[1:])
    else:
        x11 = int(x11)
##    print(x11.split("   "), type(x11))
    if x11 > bax:
        #원리/1 21 21 21 21 21 21
        if notice ==1:
            seb = time.time()
            t = seb - fsa
            upti(t,bax, "up",fullcheck)
            notice = 0
            
        if notice == 0:
            fsa = time.time()
            notice =1

    if x11 < bax:
        #원리/1 21 21 21 21 21 21
        if notice ==1:
            seb = time.time()
            t = seb - fsa
            upti(t,bax, "down",fullcheck)
            notice = 0
            
        if notice == 0:
            fsa = time.time()
            notice =1
            
    bax = x11

@app.route('/')
def main():
    if request.headers.get('accept') == 'text/event-stream':
        def events():
            while True:
                message = connection.recv(1024)
                message = message.decode()
                
                floo = message.split(" ")

                if 'full' in message:
                    if len(message) <=10:
                        timechecker(floo[-1],1)
##                        print("yy," , floo)
##                    timechecker(floo[1])
                if not 'full' in message:
                    if len(message) <=6:                        
                        timechecker(floo[1],0)
##                        print("xx", floo)
                if floo[3][0] == "1":
                    message +=  " 한가"
                if floo[3][0] == "2":
                    message +=  " 바쁨"
                if floo[3][0] == "3":
                    message +=  " FULL"
                print(message)
                
                yield "data: %s\n\n" % (message)

        return Response(events(), content_type='text/event-stream')
    return redirect(url_for('static', filename='main.html'))
if __name__ == "__main__":
    app.run(host='0.0.0.0',port='5000')
