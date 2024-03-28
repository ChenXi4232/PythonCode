import cv2
import time


# 读取视频并获取视频帧率、分辨率
cameraCapture = cv2.VideoCapture(
    "rtsp://admin:linke2023@192.168.2.102/ch1/sub/av_stream?tcp"
)
fps = cameraCapture.get(5)
size = (
    int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)


x = 10  # 水印坐标
y = 10  # 水印坐标
i = 1
step_x = 5
step_y = 5

# 读取视频第一帧
success, frame = cameraCapture.read()


# 提示停止方法
print('Showing camera. Press key "Q" to quit.')
print('Press key "S" to start recording.')
Quit = 1  # 是否继续运行标志位
Record = 0  # 录制视频标志位
flag = 0  # 是否已经开始录制标志位

cameraWriter = None  # 视频写入对象

while success and Quit:
    keycode = cv2.waitKey(1)
    if keycode & 0xFF == ord("q"):  # 如果按下“Q”键，停止运行标志位置1，跳出while循环，程序停止运行
        Quit = 0
    if keycode & 0xFF == ord("s"):  # 如果按下“S”键，开始录制摄像头视频
        Record = 1
    if keycode & 0xFF == ord("x"):  # 如果按下“X”键，停止录制摄像头视频
        Record = 0

    if Record:
        if flag == 0:
            # 生成时间戳
            now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
            vname = "D:/Py_program/" + now + r"CameraExample.avi"

            # 创建新视频
            cameraWriter = cv2.VideoWriter(
                vname, cv2.VideoWriter_fourcc("I", "4", "2", "0"), fps, size
            )
            flag = 1

        # 给图片添加水印
        # cv2.putText(frame, '', (x, y),
        #            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
        cameraWriter.write(frame)  # 给新视频添加新帧
        # 水印坐标变化
        # if x > size[0]:
        #    step_x = -5
        # if x < 0:
        #    step_x = 5
        # if y > size[1]:
        #    step_y = -5
        # if y < 0:
        #    step_y = 5
        # x += step_x
        # y += step_y
        print("第" + str(i) + "帧,")
        i = i + 1
        print('Press key "X" to end recording.')
        print("\n\t")

    cv2.imshow("frame", frame)
    success, frame = cameraCapture.read()  # 逐帧读取视频

if success == 0:  # 提示由于摄像头读取失败停止程序
    print("Camera disconnect !")
print("Quitted!")  # 提示程序已停止
# 释放摄像头
cameraCapture.release()
# 程序停止前关闭所有窗口
cv2.destroyAllWindows()
