import glob
from picamera import PiCamera
import cv2
from PIL import Image, ImageFilter
import numpy as np
import time, socket, sys

camera = PiCamera()


def define(now, k):  # a = 현재 숫자
    global beforsw, befor, fresult, fcount, bsw, arrow,stopcnt
    if beforsw == 0:
        befor = now
        fcount = k
        beforsw = 1
    if 2 <= befor <= 9 and now == 0:
        fcount = fcount + 1
    elif befor == 0 and 2 <= now <= 9:
        fcount = fcount - 1
##    print(befor, now)
    if abs(befor - now) >= 3 and now!=0 and befor !=0:
        fresult = befor
        now = befor
##        print("오바")
    else:
        if befor == now:
            stopcnt +=1
        else:
            stopcnt=0
            if  befor - now < -5 and befor==0:
                arrow = "▼"
                
            elif 3 >= befor - now >0 and now==0:
                arrow = "▼"
    

            elif befor==0 and -3 < befor - now < 0:
                arrow = "▲"
            elif now ==0 and  befor - now > 5:
                arrow = "▲"

                
            elif befor - now < 0:
                arrow = "▲"
                if k ==-1:
                    arrow = "▼"
            elif befor - now > 0:
                arrow = "▼"
                if k ==-1:
                    arrow = "▲"
        if stopcnt>=10:
            arrow= " "
        fresult = now
        
    if fcount != 0:
        fresult = str(fcount) + str(fresult)
    befor = now
    return fresult

def resize20(digitlmg):
    global mean
    img = cv2.imread(digitlmg)
    ##    gray = img
##    print("asdf",mean)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret = cv2.resize(gray, (20, 20), fx=1, fy=1, interpolation=cv2.INTER_AREA)
    mean = cv2.mean(ret)[0]
    ret, thr = cv2.threshold(ret, mean, 255, cv2.THRESH_BINARY_INV)
    ##    cv2.imshow('ret', thr)
    return thr.reshape(-1, 400).astype(np.float32)




def loadLearningDigit(ocrdata):
    with np.load(ocrdata) as f:
        traindata = f["train"]
        traindata_labels = f["train_labels"]
    return traindata, traindata_labels


def OCR_for_Digits(test, traindata, traindata_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(traindata, cv2.ml.ROW_SAMPLE, traindata_labels)
    ret, result, neighbors, dist = knn.findNearest(test, k=5)
    return result


def main():
    global pas, count, a, b1, an, uu, bsw, p, B
    numberth = 0
    ocrdata = "el.npz"
    traindata, traindata_labels = loadLearningDigit(ocrdata)
    digit = "2.jpg"
    test = resize20(digit)
    result = OCR_for_Digits(test, traindata, traindata_labels)
    result[0][0] = int(result[0][0])
    if first_setting == 1:
        p = define(result[0][0], B)
    if p == 1 or p == 2:
        B = under()
        
    if B == -1:
        p = "B" + str(p)
    else:
        p = str(p)
    return p


def imcut(iim):  # 사진 자르기
    global ttt, r, z, tt, ffuull, tttt
    img = cv2.imread(iim)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array(
        [0, 100, 101]
    )  # hsv     옛날거: [141,18,100]      이것도 됨: [0,0,100]
    hsv_upper = np.array([251, 255, 200])
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    cv2.imwrite("se11.jpg", mask)
    cuter = 0
    counter = 0
    sw = 0
    r = Image.open("se11.jpg")
    y = r.size[1]
    for j in range(0, r.size[1]):
        if sw == 1:
            break
        y = y - 1
        for i in range(0, r.size[0]):
            rgb = r.getpixel((i, y))  # i,j 위치에서의 RGB 취득
            if rgb == 255:
                if cuter == 0:
                    cuter = 1
                    tt = y + 5
                counter = 0
        if counter > 3:  # 3
            t = y
            sw = 1
            break
        if rgb == 0 and cuter == 1:
            counter = counter + 1
    x = r.size[0]
    croped = mask[t:tt, 0:x]
    croped1 = img[t:tt, 0:x]
    cv2.imwrite("33k.jpg", croped)  #'33k.jpg'
    r = Image.open("33k.jpg")  #'33k.jpg'
    counter = 0
    cuter = 0
    sw = 0
    for j in range(0, r.size[0]):
        if sw == 1:
            break
        x = x - 1
        for i in range(0, r.size[1]):
            rgb = r.getpixel((x, i))  # i,j 위치에서의 RGB 취득
            if rgb == 255:

                if cuter == 0:
                    cuter = 1
                    zz = x + 5  # x+5
                counter = 0
                # print(counter)
        if counter > 1:
            z = x
            sw = 1
            break
        if rgb == 0 and cuter == 1:
            counter = counter + 1
    croped = croped1[t:tt, z:zz]  # t :ymax , tt : ymin , z : xmax, zz :xmin
    cv2.waitKey()
    ttt = tt - t
    ffuull = ttt * 2 - 15
    ttt = ttt / 2
    ttt = t + ttt
    tttt = zz - 9  # zz-2
    
    memo0 = open("0.txt","w")
    memo0.write(str(t))
    memo0.close()
    memo1 = open("1.txt","w")
    memo1.write(str(tt))
    memo1.close()
    memo2 = open("2.txt","w")
    memo2.write(str(z))
    memo2.close()
    memo3 = open("3.txt","w")
    memo3.write(str(zz))
    memo3.close()

    memo4 = open("ttt.txt","w")
    memo4.write(str(int(ttt)))
    memo4.close()


    memo5 = open("full.txt","w")
    memo5.write(str(int(ffuull)))
    memo5.close()


    memo6 = open("tttt.txt","w")
    memo6.write(str(int(tttt)))
    memo6.close()
    return t, tt, z, zz


def full(imm):
    r = Image.open(imm)
    ##    print("x좌표 : ",tttt, "ymax좌표 : ",tt, "y시작좌표",ffuull)
    rcol = 0
    for i in range(tt + ffuull, tt + ffuull + 7):
        rgb = r.getpixel((tttt, i))
        if rgb[0] > 112:
            rcol = rcol + 1
            break
    if rcol == 1:
        return 1
    else:
        return 0


def web():
    global soc
    print("Client Server...")
    time.sleep(1)
    # Get the hostname, IP Address from socket and set Port
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    shost = socket.gethostname()
    ip = socket.gethostbyname(shost)
    # get information to connect with the server
    print(shost, "({})".format(ip))
    server_host = "175.197.167.234"  # "125.133.112.30"
    name = "client"
    port = 1234
    print("Trying to connect to the server: {}, ({})".format(server_host, port))
    time.sleep(1)
    soc.connect((server_host, port))
    print("Connected...\n")
    soc.send(name.encode())
    server_name = soc.recv(1024)
    server_name = server_name.decode()
    print("{} has joined...".format(server_name))
    print("Enter [bye] to exit.")




def fistr_check():
    global uuunder, B
    r = Image.open("1.jpg")
    for i in range(int(z) - 40, int(z) - 5):
        rgb = r.getpixel((i, ttt))
        if rgb[0] >= 100:
            B = -1
            uuunder=uuunder+1
            break
    if uuunder == 0:
        B = 0
    uuunder = 0
    return B



def under():  # 지하
    global uuunder, B
    r = Image.open("1.jpg")
    for i in range(int(z) - 40, int(z) - 5):
        rgb = r.getpixel((i, ttt))
        if rgb[0] >= 100:
            B = -1
            uuunder=uuunder+1
            break
    if uuunder == 0:
        B = 0
    uuunder = 0
    return B


c = 0.0
bsw = 0
beforsw = 0
befor = 0
B = 0
stopcnt = 0
uuunder = 0
sw = 0
lsw = 0
first_setting = 0
p = 0
arrow = ""
# try:
start = input("1. 좌표설정 없이 시작 /  2.좌표설정")
if start == "1":
    sw = 1
    memo0 = open("0.txt","r")
    t = int(memo0.read())
    memo0.close()
    
    memo1 = open("1.txt","r")
    tt = int(memo1.read())
    memo1.close()
    
    memo2 = open("2.txt","r")
    z = int(memo2.read())
    memo2.close()
    
    memo3 = open("3.txt","r")
    zz = int(memo3.read())
    memo3.close()
    
    memo4 = open("ttt.txt","r")
    ttt = int(memo4.read())
    memo4.close()
    memo5 = open("full.txt","r")
    ffuull = int(memo5.read())
    memo5.close()
    
    memo6 = open("tttt.txt","r")
    tttt = int(memo6.read())
    memo6.close()
    print(tt, ffuull,tttt)
    line = [(t),(tt),(z),(zz)]
else:
    sw = 0

    
while True:
    camera.capture('1.jpg')
    if sw == 0:
        line = imcut("1.jpg")  #'3.jpg'
        sw = 1
    elif sw == 1:
        img = cv2.imread("1.jpg")  #'3.jpg'
        croped = img[line[0] : line[1], line[2] : line[3]]
        cv2.imwrite("2.jpg", croped)  #'2.jpg'F
        go = input("1. go  / 2.return")
        if go == "1":
            if int(main()) != 1:
                sw = 2
            elif int(main()) == 1:
                sw = 0
        else:
            sw = 0
    elif sw == 2:
            web()
            sw = 3
    elif sw == 3:
        if first_setting == 1:
            img = cv2.imread("1.jpg")
            croped = img[line[0] : line[1], line[2] : line[3]]
            cv2.imwrite("2.jpg", croped)
            num = str(main())
            full_v = full("1.jpg")
            server = (
                str(full_v)
                + " "
                + str(arrow)
                + " "
                + str(num[:len(num)-2])
            )  

            soc.send(server.encode())
            
        elif first_setting == 0:
            if fistr_check()==0:
                first_setting = 1
