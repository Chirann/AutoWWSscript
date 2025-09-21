import cv2
import numpy as np
import pyautogui
import time
def get_screen():
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)



def nothing(x):
    pass

def make_circle_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 红色圈边缘
    # lower_red1 = np.array([0, 150, 150])
    # upper_red1 = np.array([50, 255, 255])
    # lower_red2 = np.array([170, 150, 150])
    # upper_red2 = np.array([179, 255, 255])
    # red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    lower_red1 = np.array([0, 180, 198])
    upper_red1 = np.array([13, 255, 255])
    lower_red2 = np.array([134, 180, 0])
    upper_red2 = np.array([179, 255, 255])
    lower_red3 = np.array([0, 104, 246])
    upper_red3 = np.array([18, 154, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)  | cv2.inRange(hsv, lower_red3, upper_red3)
    # 白色圈边缘
    # lower_white = np.array([0, 0, 180])
    # upper_white = np.array([180, 50, 255])
    # white_mask = cv2.inRange(hsv, lower_white, upper_white)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180,26, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 合并
    mask =  red_mask |white_mask
    # mask = white_mask
    # mask = red_mask

    # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_circles_by_contours(img,debug=False):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # # 筛选红色 + 白色区域（可按需求调整范围）
    # red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (160, 100, 100), (179, 255, 255))
    # white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    # mask = red_mask | white_mask

    # # 形态学操作去噪
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    mask=make_circle_mask(img)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = []
    debug_img = img.copy()
    for cnt in contours:
        # if len(cnt) < 5:
        #     continue
        # (x, y), radius = cv2.minEnclosingCircle(cnt)
        # area_cnt = cv2.contourArea(cnt)
        # area_circle = np.pi * (radius ** 2)

        # if radius < 20:  # 太小的忽略
        #     continue

        # # 圆度判断：轮廓面积 / 拟合圆面积 比值接近 1 表示比较圆
        # circularity = area_cnt / area_circle

        # if 0.6 < circularity < 1.2:  # 可调范围
        #     circles.append((int(x), int(y), int(radius)))
        #     if debug:
        #         cv2.circle(debug_img, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        #         cv2.circle(debug_img, (int(x), int(y)), 2, (0, 0, 255), 3)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (x, y), (MA, ma), angle = ellipse
            radius = (MA + ma) / 4  # 近似等效半径
            # 根据椭圆长短轴比过滤非圆
            if radius<40 or radius>220:
                continue
            if 0.80 < MA/ma < 1.3:  # 越接近1越圆
                new_circle = (int(x), int(y), int(radius))
                if x<600 or x>2200 or y>1400 or y<200:
                    continue
                if not circles:  # 如果 circles 为空，直接加
                    circles.append(new_circle)
                else:
                    merged = False
                    for i, (cx, cy, cr) in enumerate(circles):
                        dist2 = (cx - new_circle[0])**2 + (cy - new_circle[1])**2
                        if dist2 < max(cr, new_circle[2])**2:
                            # 更新为更大的圆
                            if new_circle[2] > cr:
                                circles[i] = new_circle
                            merged = True
                            break
                    if not merged:
                        circles.append(new_circle)
                if debug:
                    cv2.ellipse(debug_img, ellipse, (0,255,0), 2)
    print("original circles")
    print(circles)
    if circles:
        i=0
        while i<len(circles):
            ave=sum([x[2] for x in circles])/len(circles)
            std=np.sqrt(sum([(x[2]-ave)**2 for x in circles])/len(circles))
            if std<30:
                break
            if ave-std>circles[i][2] or circles[i][2]>ave+std:
                circles.pop(i)
            else:
                i+=1
    if debug:
        cv2.imshow("Mask", mask)
        cv2.imshow("Detected Circles", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(circles)

    return circles

def interactive_hsv_tuner(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("无法读取图片，请检查路径！")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 创建窗口
    cv2.namedWindow("Tuner")

    # 创建滑动条 (H 范围: 0-179, S: 0-255, V: 0-255)
    cv2.createTrackbar("H min", "Tuner", 0, 179, nothing)
    cv2.createTrackbar("H max", "Tuner", 179, 179, nothing)
    cv2.createTrackbar("S min", "Tuner", 0, 255, nothing)
    cv2.createTrackbar("S max", "Tuner", 255, 255, nothing)
    cv2.createTrackbar("V min", "Tuner", 0, 255, nothing)
    cv2.createTrackbar("V max", "Tuner", 255, 255, nothing)

    while True:
        # 获取滑动条值
        h_min = cv2.getTrackbarPos("H min", "Tuner")
        h_max = cv2.getTrackbarPos("H max", "Tuner")
        s_min = cv2.getTrackbarPos("S min", "Tuner")
        s_max = cv2.getTrackbarPos("S max", "Tuner")
        v_min = cv2.getTrackbarPos("V min", "Tuner")
        v_max = cv2.getTrackbarPos("V max", "Tuner")

        # 生成 mask
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # 结果可视化
        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("Original", img)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

        # 按下 q 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("最终选择范围：")
            print("lower =", lower.tolist())
            print("upper =", upper.tolist())
            break

    cv2.destroyAllWindows()

# 使用示例
# interactive_hsv_tuner("./templates/test.png")  # 替换为你的截图路径

# 使用方法
img=cv2.imread("templates/test.png")
find_circles_by_contours(img,True)  # 替换成你的截图路径
