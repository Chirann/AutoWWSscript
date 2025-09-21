import cv2
import numpy as np
import pytesseract
import pyautogui
import random
import time


def wait_until_captured(screenshot_func, region_box, timeout=60):
    """
    等待目标区域从红色/白色变为绿色
    screenshot_func: 一个函数，返回当前屏幕截图（cv2格式）
    region_box: 目标区域的(x, y, w, h)
    timeout: 最长等待时间
    """
    start = time.time()
    while time.time() - start < timeout:
        img = screenshot_func()
        x, y, w, h = region_box
        roi = img[y:y+h, x:x+w]

        # 转换到HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 定义绿色范围
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        green_ratio = np.sum(mask > 0) / (w*h)

        if green_ratio > 0.5:  # 大部分区域变成绿色
            print("区域已占领！")
            return True

        time.sleep(1)

    print("等待超时，占领检测失败")
    return False

# -------------------
# OCR配置
# -------------------
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"


# -------------------
# 识别自己位置（小箭头）
# -------------------
def find_my_position(img):
    template=cv2.imread("templates/arrow3.png", cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, template_bin = cv2.threshold(template_gray, 200, 255, cv2.THRESH_BINARY)
    template_cnt, _ = cv2.findContours(template_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    template_cnt = template_cnt[0]

    matches = []
    for cnt in contours:
        score = cv2.matchShapes(template_cnt, cnt, cv2.CONTOURS_MATCH_I1, 0.0)
        area = cv2.contourArea(cnt)
        if area < 100 or area > 500:  # 根据箭头实际大小调节
            continue
        if score < 0.05:  # 阈值调小更严格
            x, y, w, h = cv2.boundingRect(cnt)
            matches.append((x, y, w, h, score))
    
    if matches:
        # 返回最佳匹配的箭头位置
        best = min(matches, key=lambda x: x[-1])
        [print(x) for x in matches]
        return (best[0], best[1])  # 左上角坐标
    return None

# def get_stable_arrow_position(arrow_template, n_frames=10):
#     all_positions = []

#     for _ in range(n_frames):
#         frame = get_screen()  # 你自己的截图函数
#         pos = find_my_position(frame, arrow_template)
#         if pos:
#             all_positions.append(pos)
#         time.sleep(0.1)  # 适当延时，避免连续抓帧完全一样

#     if not all_positions:
#         return None

#     # 转 numpy 方便处理
#     arr = np.array(all_positions)

#     # 计算所有点的平均值
#     mean_pos = np.mean(arr, axis=0)

#     # 进一步：找和 mean 最近的一组点
#     dists = np.linalg.norm(arr - mean_pos, axis=1)
#     mask = dists < np.median(dists) * 1.2  # 筛掉离群点
#     stable_positions = arr[mask]

#     # 最终位置取平均
#     final_pos = np.mean(stable_positions, axis=0)
#     return tuple(map(int, final_pos))

# -------------------
# 识别目标区域 (A/B/C/D 且是红色/白色区域)
# -------------------
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
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    # 白色圈边缘
    # lower_white = np.array([0, 0, 180])
    # upper_white = np.array([180, 50, 255])
    # white_mask = cv2.inRange(hsv, lower_white, upper_white)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180,26, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # 合并
    mask = red_mask | white_mask
    # mask = white_mask
    # mask = red_mask

    # # 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def find_capturable_zones(img,debug=False):
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
            if (ave-std>circles[i][2] or circles[i][2]>ave+std):
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


# --------------------
# 1. 特征匹配工具函数
# --------------------
def feature_match(screen, template, threshold=0.6):
    """
    在 screen 图像中查找 template 的位置（ORB特征匹配）
    返回匹配到的位置中心点 (x,y)，若未找到返回 None
    """
    # 转灰度
    img_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # ORB特征
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None:
        return None

    # 暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return None

    # 按距离排序
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * threshold)]

    if len(good_matches) < 4:  # 匹配点太少，丢弃
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        return None

    h, w = template_gray.shape
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 取目标的中心点
    center_x = int(np.mean(dst[:, 0, 0]))
    center_y = int(np.mean(dst[:, 0, 1]))
    return (center_x, center_y)


# --------------------
# 2. 获取屏幕截图
# --------------------
def get_screen():
    screenshot = pyautogui.screenshot()
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


# --------------------
# 3. 判定是否非法点
# --------------------
def is_invalid_target(screen):
    text = pytesseract.image_to_string(screen, lang="chi_sim")  # 中文OCR
    return "无效路径" in text


# -------------------
# 在区域内随机选点
# -------------------
def random_point_in_zone(zone_center):
    x, y, radius = zone_center
    radius=int(radius*0.7)
    dx = random.randint(-radius, radius)
    dy = random.randint(-radius, radius)
    return (x + dx, y + dy)


def clickreset():
    pyautogui.click(10,10)
    return 
# --------------------
# 4. 主寻路逻辑
# --------------------
def auto_capture():
    """
    arrow_template: 自己的箭头小图
    area_templates: {"A": pathA, "B": pathB, ...} 区域模板
    """
    print("start")
    time.sleep(1)
    while True:
        try:
            print("loop start")
            numtry=0
            pyautogui.press("m")
            time.sleep(2)
            targets=[]
            while True:
                numtry+=1
                screen = get_screen()
                if not targets:
                    targets=find_capturable_zones(screen)
                else:
                    tmp=find_capturable_zones(screen)
                    if len(tmp)>len(targets):
                        targets=tmp
                # 定位自己位置
                if not targets:
                    pyautogui.press('m')
                    print("no area")
                    return
                my_pos = find_my_position(screen)
                if my_pos:
                    break
                if numtry>50:
                    print("try exceeds")
                    if targets:
                        tar=random_point_in_zone(targets[0])
                        pyautogui.click(tar[0], tar[1])
                        time.sleep(1)
                        pyautogui.click(tar[0], tar[1])
                        time.sleep(1)
                    pyautogui.press("m")
                    return
                time.sleep(0.2)
            print("mypos captured")
            print(my_pos)

            # 按距离排序，选最近的
            targets.sort(key=lambda x: (my_pos[0]-x[0])**2+(my_pos[1]-x[1])**2)
            print(targets)
            if (my_pos[0]-targets[0][0])**2+(my_pos[1]-targets[0][1])**2<1.5*targets[0][2]*2:
                targets.pop(0)
            target_pos = targets[0]
            print(f"前往最近的区域")
            # 点击地图目标
            tar=random_point_in_zone(target_pos)
            pyautogui.click(tar[0], tar[1])
            time.sleep(1)
            pyautogui.click(tar[0], tar[1])
            time.sleep(1)
            tar=random_point_in_zone(target_pos)
            pyautogui.click(tar[0], tar[1])
            time.sleep(1)
            pyautogui.click(tar[0], tar[1])
            time.sleep(1)
            tar=random_point_in_zone(target_pos)
            pyautogui.click(tar[0], tar[1])
            time.sleep(1)
            pyautogui.click(tar[0], tar[1])
            time.sleep(1)
            pyautogui.keyDown("shift")
            if len(targets)>1:
                newtag=random_point_in_zone(targets[-1])
                pyautogui.click(newtag[0], newtag[1])
                time.sleep(1)
                pyautogui.click(newtag[0], newtag[1])
            pyautogui.keyUp("shift")
            clickreset()

            # 判定是否非法点
            screen = get_screen()
            if is_invalid_target(screen):
                print("非法目标点，重试...")
                # continue

            # # 等待区域变绿（占领完成）
            # while True:
            #     screen = get_screen()
            #     # 用颜色判定：目标区域变绿则退出循环
            #     b, g, r = screen[target_pos[1], target_pos[0]]
            #     if g > r and g > b:  # 简单的绿色判定
            #         print(f"区域已占领！")
            #         break
            #     time.sleep(3)

            # 继续下一轮
            time.sleep(1)
            pyautogui.press("m")
            break
        except:
            continue
# while True:
#     zones=find_my_position(get_screen())
#     print(zones)
#     if zones:
#         pyautogui.click(zones)
#     time.sleep(2)
# auto_capture()
