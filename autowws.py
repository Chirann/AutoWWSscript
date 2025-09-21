import cv2
import numpy as np
import pyautogui
import mss
import win32gui
import win32ui
import win32con
import win32api
import win32print
import time
import random
import ctypes
from ctypes import windll , WinDLL
import capturemod
# from datetime import datetime

def get_window_rect(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        raise Exception("未找到窗口: " + window_name)
    rect = win32gui.GetWindowRect(hwnd)
    return rect

def capture_window(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        raise Exception("未找到窗口: " + window_name)

    # 创建窗口的设备上下文
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width = right - left
    height = bottom - top

    # 创建兼容的DC与位图
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # 检查窗口是否最小化
    if win32gui.IsIconic(hwnd):
        # 强制恢复窗口（如果你希望最小化时也能取图，可以去掉这个）
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    # PrintWindow 可以在最小化、后台也截到内容
    # result = win32print.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
    user32= WinDLL('user32',use_last_error=True)
    result = user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)

    # 清理
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    if result != 1:
        raise Exception("PrintWindow 捕获失败")

    # 返回与原函数相同格式
    # 注意：这里返回的是窗口客户区在屏幕上的位置
    rect = win32gui.GetWindowRect(hwnd)
    return rect

def find_template(template_path, monitor, threshold=0.8):
    with mss.mss() as sct:
        screenshot = np.array(sct.grab(monitor))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    h, w = template.shape[:2]

    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    if len(loc[0]) > 0:
        y, x = loc[0][0], loc[1][0]
        center_x = monitor["left"] + x + w // 2
        center_y = monitor["top"] + y + h // 2
        return (center_x, center_y)
    return None

def click_template(template_path, monitor, threshold=0.8):
    pos = find_template(template_path, monitor, threshold)
    if pos:
        rad=[random.uniform(0.98,1.02),random.uniform(0.98,1.02)]
        shiftpos=[pos[p]*rad[p] for p in range(len(pos))]
        pyautogui.click(shiftpos)
        print(f"点击 {template_path} at {shiftpos}")
        return True
    return False


def feature_match(screenshot, template_path, threshold=10):
    img_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, 0)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    return len(matches) >= threshold


def get_current_screen(monitor):
    with mss.mss() as sct:
        screenshot = np.array(sct.grab(monitor))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    states = {
        "main": "templates/main.png",
        "main2": "templates/main2.png",
        "loading1": "templates/loading1.png",
        "loading2": "templates/loading2.png",
        "result": "templates/result.png",
        "dead": "templates/dead.png",
        "map": "templates/map.png",
        "nextbattle1": "templates/nextbattle1.png",
        "nextbattle2": "templates/nextbattle2.png",
        "sbchoice": "templates/sbchoice.png",
        "aslain":"templates/aslain.png",
        "logout":"templates/logout.png",
        "retry" : "templates/retry.png"
    }

    for state, template_path in states.items():
        pos = find_template(template_path, monitor, threshold=0.8)
        if pos:
            return state

    return "unknown"

if __name__ == "__main__":
    window_name = "《战舰世界》"
    rect = capture_window(window_name)
    monitor = {
        "left": rect[0],
        "top": rect[1],
        "width": rect[2] - rect[0],
        "height": rect[3] - rect[1]
    }

    time.sleep(3)
    cnt=0
    n=90
    tryn=0
    while True:
        time.sleep(1)
        tmp=get_current_screen(monitor)
        print("当前窗口为"+str(tmp))
        t=time.localtime()
        if 19<t.tm_hour<23 and False:
            print("Still time block")
            time.sleep(600)
        if  find_template("buttoms/enter.png",monitor,threshold=0.8):
            tryn=0
            print("当前为main")
            if find_template("buttoms/enter.png",monitor,threshold=0.8) and find_template("buttoms/aliance.png",monitor,threshold=0.8):
            # if find_template("buttoms/enter.png",monitor,threshold=0.8):
                click_template("buttoms/enter.png", monitor)
                cnt+=1
                time.sleep(2)
            else:
                print("error")
        elif tmp=="loading1" or tmp=="loading2" or find_template("buttoms/start.png",monitor,threshold=0.8):
            tryn=0
            print("loading")
            if find_template("buttoms/start.png",monitor,threshold=0.7):
                print('start')
                click_template("buttoms/start.png", monitor)
                capturemod.clickreset()
                pyautogui.press(" ")
                time.sleep(10)
                pyautogui.press("w")
                pyautogui.press("w")
                pyautogui.press("w")
                pyautogui.press("w")
                time.sleep(40)
                pyautogui.press("u")
                pyautogui.press("a")
                pyautogui.press("d")
                lasttrig=0
                starttime=time.time()
                while True:
                    timenow=time.time()
                    tmp=get_current_screen(monitor)
                    print("当前窗口为"+str(tmp))
                    if find_template("buttoms/dead.png",monitor,threshold=0.8) or find_template("buttoms/dead2.png",monitor,threshold=0.8):
                        print("dead")
                        pyautogui.press("esc")
                        time.sleep(2)
                        break
                    elif tmp=="map":
                        print("地图")
                    elif find_template("templates/restart.png",monitor,threshold=0.8) and find_template("buttoms/restart.png",monitor,threshold=0.8):
                        print("restart")
                        click_template("buttoms/restart.png", monitor)
                        click_template("buttoms/restart.png", monitor)
                        time.sleep(60)
                        break
                    elif find_template("buttoms/admit.png",monitor,threshold=0.8):
                        print("retry")
                        click_template("buttoms/admit.png", monitor)
                        click_template("buttoms/admit.png", monitor)
                        continue
                    elif tmp=="nextbattle1" or find_template("buttoms/continue.png", monitor,threshold=0.8):
                        print("结算")
                        click_template("buttoms/continue.png", monitor)
                        click_template("buttoms/continue.png", monitor)
                        cnt+=1
                        if cnt>n:
                            print("cnt=",cnt)
                            cnt=0
                            n=random.randint(70,100)
                            time.sleep(1800)
                        time.sleep(5)
                        break
                    elif tmp=="nextbattle2" or find_template("buttoms/back.png", monitor,threshold=0.8):
                        print("退回主界面")
                        click_template("buttoms/back.png", monitor)
                        click_template("buttoms/back.png", monitor)
                        cnt+=1
                        if cnt>n:
                            print("cnt=",cnt)
                            cnt=0
                            n=random.randint(70,100)
                            time.sleep(1800)
                        time.sleep(5)
                        break     
                    elif tmp=="sbchoice" or find_template("buttoms/off.png", monitor):
                        print("关闭")
                        click_template("buttoms/off.png", monitor)
                        click_template("buttoms/off.png", monitor)
                    elif find_template("buttoms/backtogame.png", monitor):
                        print("Not logout")
                        click_template("buttoms/backtogame.png", monitor)
                        click_template("buttoms/backtogame.png", monitor)
                        time.sleep(5)
                    else:
                        # i=random.randint(0, 3)
                        # opr=["a","d","w","s"]
                        # if i<2:
                        #     j=random.randint(1,4)
                        #     pyautogui.keyDown(oprs[i])
                        #     time.sleep(j)
                        #     pyautogui.keyUp(opr[i])
                        #     time.sleep(5)
                        #     continue
                        # pyautogui.press(opr[i])
                        if lasttrig==0 or timenow-lasttrig>=300:
                            print('ready for capture')
                            capturemod.auto_capture()
                            if lasttrig!=0:
                                pyautogui.press("r")
                                pyautogui.press("t")
                                pyautogui.press("u")
                            lasttrig=timenow
                            continue
                        elif timenow-starttime>60*20:
                            break
                        else:
                            print("wait till ends")
                            time.sleep(5)
                            continue
        elif find_template("buttoms/admit.png",monitor,threshold=0.8):
            print("retry")
            click_template("buttoms/admit.png", monitor)
            click_template("buttoms/admit.png", monitor)
            continue
        elif find_template("buttoms/dead.png",monitor,threshold=0.8) or find_template("buttoms/dead2.png",monitor,threshold=0.8):
            print("dead")
            pyautogui.press("esc")
            time.sleep(5)
            tryn=0
        elif tmp=="map":
            print("地图")
        elif find_template("templates/restart.png",monitor,threshold=0.8):
            print("restart")
            click_template("buttoms/restart.png", monitor)
            click_template("buttoms/restart.png", monitor)
            time.sleep(60)
            break
        elif tmp=="nextbattle":
            print("下一把")
            if cnt>n:
                print("cnt=",cnt)
                cnt=0
                n=random.randint(70,100)
                time.sleep(1800)
            click_template("buttoms/continue.png", monitor)
            time.sleep(5)
            tryn=0
        elif tmp=="nextbattle2" or find_template("buttoms/back.png", monitor,threshold=0.8):
            print("退回主界面")
            click_template("buttoms/back.png", monitor)
            click_template("buttoms/back.png", monitor)
            cnt+=1
            if cnt>n:
                print("cnt=",cnt)
                cnt=0
                n=random.randint(70,100)
                time.sleep(1800)
            time.sleep(5)
            tryn=0
        elif find_template("buttoms/continue.png", monitor,threshold=0.8):
            print("结算")
            if cnt>n:
                print("cnt=",cnt)
                cnt=0
                n=random.randint(70,100)
                time.sleep(1800)
            click_template("buttoms/continue.png", monitor)
            click_template("buttoms/continue.png", monitor)
            tryn=0
        elif tmp=="sbchoice":
            print("关闭")
            click_template("buttoms/off.png", monitor)
            click_template("buttoms/off.png", monitor)
            time.sleep(5)
            tryn=0
        elif find_template("buttoms/aslainout.png", monitor,threshold=0.8):
            print("aslain jumped out")
            click_template("buttoms/aslainout.png", monitor)
            click_template("buttoms/aslainout.png", monitor)
            time.sleep(5)
            tryn=0
        elif tmp=="logout" and find_template("buttoms/backtogame.png", monitor):
            print("Not logout")
            click_template("buttoms/backtogame.png", monitor)
            click_template("buttoms/backtogame.png", monitor)
            time.sleep(5)
            tryn=0
        elif tryn>20:
            print('try space')
            time.sleep(2)
            tryn=0
            pyautogui.press(' ')
            time.sleep(5)
            pyautogui.press(' ')
            if get_current_screen(monitor)=="unkown":
                pyautogui.press('esc')
        else:
            time.sleep(2)
            tryn+=1