import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    img_rgb = color_img[:, :, ::-1]

    plt.subplot(2, 4, pos)
    plt.imshow(img_rgb)
    plt.title(title, fontsize=8)
    plt.axis('off')


def color_space_demo():
    # cv.namedWindow('input image', cv.WINDOW_NORMAL)  # 创建一个大小可调的窗口，初始设定函数标签是cv2.WINDOW_AUTOSIZE
    img = cv.imread("D:/python/practice/Scarlett.jpg", cv.IMREAD_COLOR)  # 读取图片
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    luv = cv.cvtColor(img, cv.COLOR_BGR2LUV)

    show_with_matplotlib(img, 'input_image', 1)
    # show_with_matplotlib(gray, 'gray_image', 2)
    show_with_matplotlib(hsv, 'hsv_image', 3)
    show_with_matplotlib(lab, 'lab_image', 4)
    show_with_matplotlib(luv, 'luv_image', 1 + 4)
    plt.show()

    '''cv.imwrite("test.jpg", img)'''  # 保存图片

    # cv.waitKey(0)  # 保持窗口
    # cv.destroyAllWindows()


def mat_demo():
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    image = cv.imread('Scarlett.jpg')
    print(image.shape)  # 高度，宽度，通道
    roi = image[100:400, 100:400, :]
    blank = np.zeros_like(image)  # 创建空白图像
    #   h,w,c = image.shape
    #   blank = np.zeros((h,w,c), dtype = np.uint8)
    blank[100:400, 100:400, :] = image[100:400, 100:400, :]
    #   blank = np.copy(image) or blank = image
    cv.imshow('image', image)
    cv.imshow('roi', roi)
    cv.imshow('blank', blank)
    cv.waitKey(0)
    cv.destroyAllWindows()


def pixel_demo():
    image = cv.imread('Scarlett.jpg')
    cv.imshow("input", image)
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            b, g, r = image[row, col]
            image[row, col] = (255 - b, 255 - g, 255 - r)
    cv.imshow('result', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def math_demo():
    image = cv.imread('Scarlett.jpg')
    # h, w, c = image.shape
    blank = np.zeros_like(image)
    blank[:, :] = (2, 2, 2)
    add_result = cv.add(image, blank)
    subtract_result = cv.multiply(image, blank)  # 加减乘除都可以

    cv.imshow('blank', blank)
    cv.imshow('input_image', image)
    cv.imshow('add_result', add_result)
    cv.imshow('subtract_result', subtract_result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def nothing(x):
    print(x)


def trackbar_demo():  # 滑杆调节亮度
    image = cv.imread('myself.jpg')
    cv.namedWindow("input_img", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("lightness", "input_img", 0, 100, nothing)
    cv.imshow("input_img", image)
    blank = np.zeros_like(image)
    while True:
        pos = cv.getTrackbarPos("lightness", "input_img")
        blank[:, :] = (pos, pos, pos)
        add_result = cv.add(image, blank)
        # subtract_result = cv.multiply(image, blank)  # 加减乘除都可以
        # cv.imshow('blank', blank)
        cv.imshow('add_result', add_result)
        c = cv.waitKey(1)
        if c == 27:
            break

    cv.destroyAllWindows()


def adjust_contrast_demo():  # trackbar调节亮度 与对比度

    image = cv.imread('myself.jpg')
    image_1 = cv.imread('Scarlett.jpg')
    roi_1 = image[0:1000, 0:1000, :]
    roi_2 = image_1[0:1000, 0:1000, :]
    cv.namedWindow("input_img", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("lightness", "input_img", 0, 100, nothing)
    cv.createTrackbar("contrast", "input_img", 100, 200, nothing)
    cv.imshow("input_img", image)
    # blank = np.zeros_like(image)
    while True:
        light = cv.getTrackbarPos("lightness", "input_img")
        contrast = cv.getTrackbarPos("contrast", "input_img") / 100
        # blank[:, :] = (light, light, light)
        add_result = cv.addWeighted(roi_1, contrast, roi_2, 1, light)
        cv.imshow('add_result', add_result)
        c = cv.waitKey(1)
        if c == 27:
            break

    cv.destroyAllWindows()


def color_table_demo():
    colormap_1 = [
        cv.COLORMAP_AUTUMN,
        cv.COLORMAP_BONE,
        cv.COLORMAP_JET,
        cv.COLORMAP_WINTER,
        cv.COLORMAP_RAINBOW,
        cv.COLORMAP_OCEAN,
        cv.COLORMAP_SUMMER,
        cv.COLORMAP_SPRING,
        cv.COLORMAP_COOL,
        cv.COLORMAP_PINK,
        cv.COLORMAP_HOT,
        cv.COLORMAP_PARULA,
        cv.COLORMAP_MAGMA,
        cv.COLORMAP_INFERNO,
        cv.COLORMAP_PLASMA,
        cv.COLORMAP_VIRIDIS,
        cv.COLORMAP_CIVIDIS,
        cv.COLORMAP_TWILIGHT,
        cv.COLORMAP_TWILIGHT_SHIFTED
    ]

    image = cv.imread("Scarlett.jpg")
    cv.namedWindow("input_img", cv.WINDOW_AUTOSIZE)
    cv.imshow("input_img", image)
    index = 0
    while True:
        dst = cv.applyColorMap(image, colormap_1[index % 19])
        index += 1
        cv.imshow("color style", dst)
        c = cv.waitKey(500)
        if c == 27:
            break
    cv.destroyAllWindows()


def bitwise_demo():
    b1 = np.zeros((400, 400, 3), dtype=np.uint8)
    b1[:, :] = (255, 0, 255)
    b2 = np.zeros((400, 400, 3), dtype=np.uint8)
    b2[:, :] = (0, 255, 0)
    cv.imshow("b1", b1)
    cv.imshow("b2", b2)

    dst1 = cv.bitwise_and(b1, b2)
    dst2 = cv.bitwise_or(b1, b2)
    dst3 = cv.bitwise_not(b1)
    cv.imshow("bitwise_and", dst1)
    cv.imshow("bitwise_or", dst2)
    cv.imshow("b1_bitwise_not", dst3)
    cv.waitKey(0)
    cv.destroyAllWindows()


def color_background_demo():  # 抠图，换背景
    b1 = cv.imread("myself.jpg")
    cv.imshow("input_img", b1)
    hsv = cv.cvtColor(b1, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    mask = cv.inRange(hsv, (78, 43, 46), (124, 255, 225))
    mask_1 = cv.bitwise_not(mask, mask)
    cv.bitwise_not(mask, mask)
    result = cv.bitwise_and(b1, b1, mask=mask)

    blank = np.zeros_like(result)
    cv.bitwise_not(blank, blank)
    cv.imshow('blank', blank)
    cv.namedWindow("input_img", cv.WINDOW_AUTOSIZE)
    cv.createTrackbar("lightness", "input_img", 0, 100, nothing)
    cv.createTrackbar("contrast", "input_img", 100, 200, nothing)
    while True:
        light = cv.getTrackbarPos("lightness", "input_img")
        contrast = cv.getTrackbarPos("contrast", "input_img") / 100
        add_result = cv.addWeighted(blank, contrast, result, 0.5, light)
        cv.imshow('add_result', add_result)

        c = cv.waitKey(1)
        if c == 27:
            break

    cv.imshow("mask", mask)
    cv.imshow("result", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def pixel_stat_demo():
    b1 = cv.imread("dog.jpg")
    print(b1.shape)
    print(np.max(b1[::2]))  # 通道最大值
    cv.imshow("input_img", b1)
    means, dev = cv.meanStdDev(b1)  # 均值与方差
    print(means, "dev: ", dev)
    cv.waitKey(0)
    cv.destroyAllWindows()


def drawing_demo():
    # b1 = np.zeros((512, 512, 3), dtype=np.uint8)
    b1 = cv.imread("Scarlett.jpg")
    cv.rectangle(b1, (200, 200), (1200, 1200), (0, 0, 255), 2, 8, 0)  # 线宽小于0表示填充
    cv.line(b1, (50, 50), (400, 400), (0, 255, 255), 2, 8, 0)
    cv.circle(b1, (200, 200), 100, (255, 255, 255), 2, 8, 0)
    cv.putText(b1, "scarlett", (200, 200), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, 8)
    cv.imshow("input_img", b1)
    cv.waitKey(0)
    cv.destroyAllWindows()


def random_color_demo():
    b1 = np.zeros((512, 512, 3), dtype=np.uint8)
    while True:
        x1 = np.random.randint(0, 512, 2, dtype=int)
        y1 = np.random.randint(0, 512, 2, dtype=int)
        bgr = np.random.randint(0, 255, 3, dtype=np.uint8)

        cv.line(b1, (x1[0], y1[0]), (x1[1], y1[1]), (int(bgr[0]), int(bgr[1]), int(bgr[2])), 1, 8, 0)
        cv.imshow("input_img", b1)
        c = cv.waitKey(100)
        if c == 27:
            break
    cv.destroyAllWindows()


def polyline_drawing_demo():
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    pts = np.array([[100, 100], [350, 100], [450, 280], [320, 450], [80, 400]], dtype=np.int32)
    # cv.fillPoly(canvas, [pts], (255, 0, 255), 8, 0)
    # cv.polylines(canvas, [pts], True, (255, 0, 255), 8, 0)
    cv.drawContours(canvas, [pts], -1, (255, 0, 0), -1)  # 第三个参数小于零表示画全部轮廓，第五个参数小于零表示填充
    cv.imshow("polyline_img", canvas)
    cv.waitKey(0)
    cv.destroyAllWindows()


'''
b1 = cv.imread("Scarlett.jpg")
b2 = b1.copy()
x1 = -1
x2 = -1
y1 = -1
y2 = -1



def mouse_drawing(event, x, y, flags, param):
    global x1, x2, y1, y2
    if event == cv.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
    if event == cv.EVENT_MOUSEMOVE:
        if x1 < 0 or y1 < 0:
            return
        x2 = x
        y2 = y
        dx = x2 - x1
        dy = y2 - y1
        if dx != 0 and dy != 0:
            b1[:, :, :] = b2[:, :, :]
            cv.line(b1, (x1, y1), (x2, y2), (0, 255, 255), 1, 8, 0)
            # cv.rectangle(b1, (x1, y1), (x2, y2), (0, 255, 255), 2, 8, 0)
    if event == cv.EVENT_LBUTTONUP:
        x2 = x
        y2 = y
        dx = x2 - x1
        dy = y2 - y1
        if dx != 0 and dy != 0:
            b1[:, :, :] = b2[:, :, :]
            cv.line(b1, (x1, y1), (x2, y2), (0, 255, 255), 1, 8, 0)
            # cv.rectangle(b1, (x1, y1), (x2, y2), (0, 255, 255), 2, 8, 0)
        x1 = -1
        x2 = -1
        y1 = -1
        y2 = -1


def mouse_demo():
    cv.namedWindow("mouse_demo", cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("mouse_demo", mouse_drawing)
    while True:
        cv.imshow("mouse_demo", b1)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()
    
'''


def norm_demo():
    image = cv.imread("Scarlett.jpg")
    cv.namedWindow("norm_demo", cv.WINDOW_AUTOSIZE)
    result = np.zeros_like(np.float32(image))
    cv.normalize(np.float32(image), result, 0, 1, cv.NORM_MINMAX, detype=cv.CV_32F)  # 归一化，浮点型显示
    cv.imshow("norm_demo", np.uint8(result * 255))  # 转换成整数形式
    print(image / 255.0)  # 可以显示浮点数和整数型
    cv.waitKey(0)
    cv.destroyAllWindows()


def resize_demo():
    image = cv.imread("Scarlett.jpg")
    h, w, c = image.shape
    cv.namedWindow("resize_demo", cv.WINDOW_AUTOSIZE)
    dst_1 = cv.resize(image, (w // 2, h // 2), interpolation=cv.INTER_CUBIC)
    dst = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
    cv.imshow("resize_demo", dst)
    cv.imshow("resize_img", dst_1)
    cv.waitKey(0)
    cv.destroyAllWindows()


def flip_demo():
    image = cv.imread("Scarlett.jpg")
    # cv.imshow("input_img", image)
    # cv.namedWindow("flip_demo", cv.WINDOW_AUTOSIZE)
    dst_1 = cv.flip(image, -1)  # 上下左右翻转
    dst_2 = cv.flip(image, 0)  # 上下翻转
    dst_3 = cv.flip(image, 1)  # 左右翻转

    show_with_matplotlib(image, 'input_image', 1)
    show_with_matplotlib(dst_1, '-1_image', 2)
    show_with_matplotlib(dst_2, '0_image', 3)
    show_with_matplotlib(dst_3, '1_image', 4)
    plt.show()


def spinning_demo():
    image = cv.imread("D:/python/practice/picture/calss.jpg")
    h, w, c = image.shape
    M = np.zeros((2, 3), dtype=np.float32)
    alpha = np.cos(np.pi / 2.0)  # 旋转角度
    beta = np.sin(np.pi / 2.0)
    print("alpha", alpha)

    #  旋转矩阵
    M[0, 0] = alpha
    M[1, 1] = alpha
    M[0, 1] = beta
    M[1, 0] = -beta
    cx = w / 2
    cy = h / 2
    tx = (1 - alpha) * cx - beta * cy
    ty = (1 - alpha) * cy + beta * cx
    M[0, 2] = tx
    M[1, 2] = ty

    # 改变旋转中心
    bound_w = int(h * np.abs(beta) + w * np.abs(alpha))
    bound_h = int(h * np.abs(alpha) + w * np.abs(beta))
    # 中心位置迁移
    M[0, 2] += bound_w / 2 - cx
    M[1, 2] += bound_h / 2 - cy

    dst = cv.warpAffine(image, M, (w, h))
    # dst_1 = cv.rotate(image, rotateCode=90, dst=None)
    N = cv.getRotationMatrix2D((w / 2.0, h / 2.0), 45, 0.5)
    dst_2 = cv.warpAffine(image, N, (w, h))
    cv.imshow("spinning_center_demo", dst)
    cv.imshow("warpAffine_image", dst_2)
    # cv.imshow("rotate_image", dst_1)
    cv.waitKey(0)
    cv.destroyWindow()


def video_demo():
    cap = cv.VideoCapture("sample.mp4")  # 括号内为视频文件路径则播放视频
    w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CAP_PROP_FPS)
    # out = cv.VideoWriter("D:/python/kingdom.mp4", cv.VideoWriter.fourcc('P', 'I', 'M', '1'), fps, (int(w), int(h)),
    # True)
    out = cv.VideoWriter("D:/python/kingdom.mp4", cv.CAP_ANY, int(cap.get(cv.CAP_PROP_FOURCC)), fps, (int(w), int(h)),
                         True)
    # 没有音频
    while True:
        ret, frame = cap.read()
        # frame = cv.flip(frame, 1)
        if ret is True:
            cv.imshow("frame", frame)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            cv.imshow("hsv_result", hsv)
            out.write(hsv)
            c = cv.waitKey(10)
        if c == 27:
            break
    cv.destroyAllWindows()

    out.release()
    cap.release()


def image_hist_demo():
    src = cv.imread("D:/python/practice/picture/Scarlett.jpg")
    cv.imshow("input_image", src)
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([src], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()
    cv.waitKey(0)
    cv.destroyWindow()


def image_hist2d_demo():
    image = cv.imread("D:/python/practice/picture/Scarlett.jpg")
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [48, 48], [0, 180, 0, 256])
    dst = cv.resize(hist, (400, 400))
    cv.normalize(dst, dst, 0, 255, cv.NORM_MINMAX)
    cv.imshow("image", image)
    dst = cv.applyColorMap(np.uint8(dst), cv.COLORMAP_JET)
    cv.imshow("hist", dst)
    plt.imshow(hist, interpolation='nearest')
    plt.title('2D Histogram')
    plt.show()
    cv.waitKey(0)
    cv.destroyWindow()


def equalizeHist_demo():  # 直方图均衡化
    image = cv.imread("D:/python/practice/picture/Scarlett.jpg", cv.IMREAD_GRAYSCALE)
    dst = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    # cv.imshow("input_image", dst)
    result = cv.equalizeHist(dst)  # 全局均衡
    res = np.hstack((dst, result))
    cv.imshow('result_image', res)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 自适应直方图均衡
    res_clahe = clahe.apply(dst)
    res_1 = np.hstack((dst, res_clahe))
    cv.imshow("res_1_image", res_1)
    cv.waitKey(0)
    cv.destroyWindow()


def blur_demo():  # 图像卷积、图像模糊、平滑滤波等
    image = cv.imread("Scarlett.jpg")
    cv.imshow("input_image", image)
    result = cv.blur(image, (3, 3))
    cv.imshow("result_image", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def conv_demo():
    image = cv.imread("Scarlett.jpg")
    cv.imshow("input_image", image)
    result_1 = cv.blur(image, (3, 3))
    result_2 = cv.GaussianBlur(image, (0, 0), 3)  # ksize必须是奇数
    cv.imshow("blur_image", result_1)
    cv.imshow("GaussianBlur_image", result_2)
    cv.waitKey(0)
    cv.destroyAllWindows()


def bfilter_demo():
    image = cv.imread("Scarlett.jpg")
    cv.imshow("input_image", image)
    result = cv.bilateralFilter(image, 0, 100, 10)
    cv.imshow("result_image", result)
    cv.waitKey(0)
    cv.destroyAllWindows()


config_text = "D:/python/Open CV/opencv_tutorial_data/models/face_detector/opencv_face_detector.pbtxt"
model_bin = "D:/python/Open CV/opencv_tutorial_data/models/face_detector/opencv_face_detector_uint8.pb"


def face_detection_demo():
    net = cv.dnn.readNetFromTensorflow(model_bin, config=config_text)
    cap = cv.VideoCapture("sample.mp4")  # 括号内为视频文件路径则播放视频
    # 人脸检测
    while True:
        ret, frame = cap.read()
        h, w, c = frame.shape
        if ret is not True:
            break
        # NCHW
        blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        outs = net.forward()  # 1*1*N*7

        # 绘制矩形
        for detection in outs[0, 0, :, :]:
            score = float(detection[2])
            if score > 0.5:
                left = detection[3] * w
                top = detection[4] * h
                right = detection[5] * w
                bottom = detection[6] * h
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2, 8, 0)  # 线宽小于0表示填充

        cv.imshow("frame", frame)
        c = cv.waitKey(1)
        if c == 27:
            break
    cv.destroyAllWindows()

    cap.release()


def dilate_erosion_demo():
    image = cv.imread("D:/python/practice/picture/myself.jpg")
    cv.imshow("image", image)
    kernel = np.ones((3, 3), np.uint8)  # 卷积核
    dilate = cv.dilate(image, kernel, iterations=5)  # 膨胀
    erosion = cv.erode(image, kernel, iterations=5)  # 腐蚀
    gradient = cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)
    #  梯度运算（膨胀-腐蚀）
    #  开操作：先腐蚀后膨胀 ，闭操作：先膨胀后腐蚀

    #  x,y方向使用sobel算子计算图像梯度，并取绝对值
    #  可以分别计算x，y方向的梯度，也可以合并一起计算，中间两个参数都改成1即可,效果hin不好
    sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
    sobelx = cv.convertScaleAbs(sobelx)
    sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
    sobely = cv.convertScaleAbs(sobely)

    together_sobelxy = cv.Sobel(image, cv.CV_64F, 1, 1, ksize=3)
    together_sobelxy = cv.convertScaleAbs(together_sobelxy)
    cv.imshow("together_sobelxy", together_sobelxy)

    #  将计算好的x，y方向的图像梯度进行加权
    sobelxy = cv.addWeighted(sobelx, 1, sobely, 1, 0)
    #  sobelxy = cv.convertScaleAbs(sobelxy)

    cv.imshow("sobelxy", sobelxy)

    res = np.hstack((dilate, erosion))  # 图像连接
    show_with_matplotlib(dilate, 'dilate', 1)
    show_with_matplotlib(erosion, 'erosion', 2)
    show_with_matplotlib(gradient, 'gradient', 3)
    show_with_matplotlib(sobelx, 'sobelx', 4)
    show_with_matplotlib(sobely, 'sobely', 1 + 4)
    show_with_matplotlib(sobelxy, 'sobelxy', 2 + 4)
    show_with_matplotlib(together_sobelxy, 'together_sobelxy', 3 + 4)
    plt.show()
    cv.waitKey(0)
    cv.destroyWindow()


def algorithm_comparsion_demo():
    image = cv.imread("D:/python/practice/picture/myself.jpg")
    dst = cv.resize(image, None, fx=0.4, fy=0.4, interpolation=cv.INTER_CUBIC)

    sobelx = cv.Sobel(dst, cv.CV_64F, 1, 0, ksize=3)
    sobelx = cv.convertScaleAbs(sobelx)
    sobely = cv.Sobel(dst, cv.CV_64F, 0, 1, ksize=3)
    sobely = cv.convertScaleAbs(sobely)
    sobelxy = cv.addWeighted(sobelx, 1, sobely, 1, 0)

    scharrx = cv.Scharr(dst, cv.CV_64F, 1, 0)
    scharry = cv.Scharr(dst, cv.CV_64F, 0, 1)
    scharrx = cv.convertScaleAbs(scharrx)
    scharry = cv.convertScaleAbs(scharry)
    scharrxy = cv.addWeighted(scharrx, 1, scharry, 1, 0)

    laplacian = cv.Laplacian(dst, cv.CV_64F)
    laplacian = cv.convertScaleAbs(laplacian)

    res = np.hstack((sobelxy, scharrxy, laplacian))

    cv.imshow("res", res)
    cv.waitKey(0)
    cv.destroyWindow()


def canny_demo():
    image = cv.imread("D:/python/practice/picture/myself.jpg")
    dst = cv.resize(image, None, fx=0.4, fy=0.4, interpolation=cv.INTER_CUBIC)

    v1 = cv.Canny(dst, 100, 250)
    v2 = cv.Canny(dst, 50, 120)

    res = np.hstack((v1, v2))

    cv.imshow("res", res)
    cv.waitKey(0)
    cv.destroyWindow()


def pyramid_demo():
    image = cv.imread("D:/python/practice/picture/myself.jpg")
    up = cv.pyrUp(image)  # 向上采样（图像变大）
    down = cv.pyrDown(image)  # 向下采样（图像变小）
    up_down = cv.pyrDown(up)  # 图像部分信息丢失，变模糊
    l_l = image - up_down  # 拉普拉斯金字塔 1、输入图像；2、缩小尺寸；3、放大尺寸；4、图像相减

    cv.imshow("l_l", l_l)

    res = np.hstack((up_down, image))
    cv.imshow("down", down)
    cv.imshow("res", res)

    cv.waitKey(0)
    cv.destroyWindow()


def contour_detection():
    image = cv.imread("D:/python/practice/picture/myself.jpg")
    dst = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    gray_image = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray_image, 127, 255, cv.THRESH_BINARY)  # threshold处理的图像必须是灰度图，其中127为阈值，thresh表示输出的二值图像
    cv.imshow("thresh", thresh)  # 255为当像素值高于或低于阈值时取的值， 最后一个参数为阈值的处理方法

    contour, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                         cv.CHAIN_APPROX_NONE)  # findcontours只返回两个值，输入可以是灰度图，但是一般采用二值图像

    draw_image = dst.copy()  # 不copy的话，原图会改变
    res = cv.drawContours(draw_image, contour, -1, (0, 0, 255), 2)
    # -1表示画出所有轮廓
    cv.imshow("res", res)

    # 轮廓特征
    cnt = contour[0]
    print(cv.contourArea(cnt))  # 计算面积
    print(cv.arcLength(cnt, True))  # 计算周长

    # 轮廓近似
    esplion = 0.001 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, esplion, True)
    draw_img = dst.copy()
    res_approx = cv.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
    cv.imshow('res_approx', res_approx)

    # 边界矩形
    dst_1 = dst.copy()
    x, y, w, h = cv.boundingRect(cnt)
    dst_1 = cv.rectangle(dst_1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.imshow('dst_1', dst_1)

    cv.waitKey(0)
    cv.destroyWindow()


def template_matching_demo():
    image = cv.imread("D:/python/practice/picture/myself.jpg")
    dst = image[500:1000, 500:1000]
    w, h, c = dst.shape
    # cv.imshow('dst', dst)
    # cv.waitKey(0)
    # cv.destroyWindow()

    methods = ['cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED', 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED']
    # 归一化的方法，效果好一点
    # res = cv.matchTemplate(image, dst, cv.TM_SQDIFF_NORMED)
    # cv.imshow('res', res)
    for meth in methods:
        image2 = cv.cvtColor(image.copy(), cv.COLOR_BGR2RGB)  # matlab图像为rgb格式，opencv图像格式为bgr

        method = eval(meth)
        print(method)
        res = cv.matchTemplate(image, dst, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        if method in [cv.TM_SQDIFF_NORMED, cv.TM_SQDIFF]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv.rectangle(image, top_left, bottom_right, 255, 2)

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title("res")
        plt.subplot(122), plt.imshow(image2, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title(meth)
        plt.show()


def fourier_demo():  # 傅里叶变换
    image = cv.imread("D:/python/practice/picture/myself.jpg", 0)
    image_float32 = np.float32(image)  # opencv要求必须是float32格式

    dft = cv.dft(image_float32, flags=cv.DFT_COMPLEX_OUTPUT)  # 傅里叶变化
    dft_shift = np.fft.fftshift(dft)  # 将低频放到中心
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    # dft得到双通道实部与虚部，需要转换成图像格式（0：255）

    rows, cols = image.shape
    crow, tcol = int(rows / 2), int(cols / 2)  # 中心位置

    # 低通滤波
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, tcol - 30:tcol + 30] = 1
    # IDFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)  # 将低频信号还回去
    image_back_d = cv.idft(f_ishift)
    image_back_d = cv.magnitude(image_back_d[:, :, 0], image_back_d[:, :, 0])

    # 高频滤波
    mask = np.ones((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, tcol - 30:tcol + 30] = 0
    # IDFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)  # 将低频信号还回去
    image_back_g = cv.idft(f_ishift)
    image_back_g = cv.magnitude(image_back_g[:, :, 0], image_back_g[:, :, 0])

    plt.subplot(141), plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title("input_image")
    plt.subplot(142), plt.imshow(image_back_d, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title("image_back_d")
    plt.subplot(143), plt.imshow(image_back_g, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title("image_back_g")
    plt.subplot(144), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.title("magnitude_spectrum")
    plt.show()


def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cornerharris_demo():
    image = cv.imread('D:/python/practice/picture/myself.jpg')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv_show('dst', image)
    """
        角点也是处在一个无论框框往哪边移动　框框内像素值都会变化很大的情况而定下来的点
        cv2.cornerHarris() 
         img - 数据类型为 float32 的输入图像。
         blockSize - 角点检测中要考虑的领域大小。
         ksize - Sobel 求导中使用的窗口大小
         k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06].
     
    """


def sift_demo():
    dst = cv.imread('D:/python/practice/picture/Scarlett.jpg')
    image1 = cv.resize(dst, (500, 500))
    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    image2 = image1[1:250, :250]
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(gray1, None)
    image = cv.drawKeypoints(gray1, kp, image1)
    cv_show('drawkeypoints', image)

    kp1, des1 = sift.compute(gray1, kp)
    kp2, des2 = sift.compute(gray2, kp)

    # bf = cv.BFMatcher(crossCheck=True)  一对一关键点匹配
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # image3 = cv.drawMatches(gray1, kp1, gray2, kp2, matches[:10], None, flags=2)
    # image3 = cv.drawMatches(gray1, kp1, gray2, kp2, sorted[:10], None, flags=2)

    # 一对多（k）匹配
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    # 过滤方法，可根据需求更改
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    image3 = cv.drawMatchesKnn(gray1, kp1, gray2, kp2, good[:10], None, flags=2)
    cv_show('image3', image3)


if __name__ == '__main__':
    sift_demo()
