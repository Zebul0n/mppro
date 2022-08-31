import re
import cv2
import mediapipe as mp
import os

images_file = "./dataset/images"  # 图片文件夹
labels_file = "./dataset/labels"  # 标签文件夹

# 节点
inode_dict = {
    0: 'NOSE',
    1: 'LEFT_EYE_INNER',
    2: 'LEFT_EYE',
    3: 'LEFT_EYE_OUTER',
    4: 'RIGHT_EYE_INNER',
    5: 'RIGHT_EYE',
    6: 'RIGHT_EYE_OUTER',
    7: 'LEFT_EAR',
    8: 'RIGHT_EAR',
    9: 'MOUTH_LEFT',
    10: 'MOUTH_RIGHT',
    11: 'LEFT_SHOULDER',
    12: 'RIGHT_SHOULDER',
    13: 'LEFT_ELBOW',
    14: 'RIGHT_ELBOW',
    15: 'LEFT_WRIST',
    16: 'RIGHT_WRIST',
    17: 'LEFT_PINKY',
    18: 'RIGHT_PINKY',
    19: 'LEFT_INDEX',
    20: 'RIGHT_INDEX',
    21: 'LEFT_THUMB',
    22: 'RIGHT_THUMB',
    23: 'LEFT_HIP',
    24: 'RIGHT_HIP',
    25: 'LEFT_KNEE',
    26: 'RIGHT_KNEE',
    27: 'LEFT_ANKLE',
    28: 'RIGHT_ANKLE',
    29: 'LEFT_HEEL',
    30: 'RIGHT_HEEL',
    31: 'LEFT_FOOT_INDEX',
    32: 'RIGHT_FOOT_INDEX'
}


class MpDetect:
    def __init__(self,
                 minDetCf=0.5):
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        if abs(minDetCf) >= 1:
            self.min_detection_confidence = 1
        else:
            self.min_detection_confidence = abs(minDetCf)  # 置信度(0~1)

    def detectImg(self, imgDir, imgOutputDir):
        IMAGE_FILES = os.listdir(imgDir)
        with self.mpPose.Pose(
                static_image_mode=True,
                min_detection_confidence=self.min_detection_confidence
        ) as pose:
            for idx, file in enumerate(IMAGE_FILES):
                file_path = imgDir + '\\' + file
                # print(file)
                img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR)
                h, w, channel = img.shape  # 获取图片的高度和宽度
                cvImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = pose.process(cvImg)
                # 没有检测到有节点:
                if not res.pose_landmarks:
                    continue

                ############################################################################################
                # 测试打印
                # print(
                #     f'Nose coordinates: ('
                #     f'{res.pose_landmarks.landmark[self.mpPose.PoseLandmark.NOSE].x * w}, '
                #     f'{res.pose_landmarks.landmark[self.mpPose.PoseLandmark.NOSE].y * h})'  # 打印鼻子的坐标
                # )
                # print('{}'.format(res.pose_landmarks).split('landmark'))  # 打印全部的节点坐标
                # print(MpDetect.getRangeCoordinate(res.pose_landmarks, img_width=w, img_height=h))
                #############################################################################################

                xmin, ymin, xmax, ymax = MpDetect.getRangeCoordinate(res.pose_landmarks, img_width=w, img_height=h)
                pt1 = (int(xmin) - 20, int(ymin) - 20)
                pt2 = (int(xmax) + 20, int(ymax) + 20)
                # print(pt1, pt2)
                cv2.rectangle(img, pt1, pt2, (0, 0, 255), 5)

                # 将节点连接起来
                self.mpDraw.draw_landmarks(img, res.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                img_output_name = imgOutputDir + '\\' + file
                cv2.imwrite(img_output_name, img)
                # 以哈希表的形式返回每个节点的比例位置
                return MpDetect.getCoordinates(res.pose_landmarks, index=-1)

    # TODO 来自摄像头的输入处理函数     ——2022/7/17
    def detectCam(self, capNum, imageOutput=False):
        count = 0
        cap = cv2.VideoCapture(capNum)
        pose = self.mpPose.Pose()
        while cap.isOpened():
            rvt, frame = cap.read()
            if rvt:
                h, w, c = frame.shape
                frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(frameRGB)
                if res.pose_landmarks:
                    print(MpDetect.getCoordinates(res.pose_landmarks, index=-1, default=count,
                                                  txtOut=True))  # 将节点都打印出来,可去掉该行代码
                    self.mpDraw.draw_landmarks(frame, res.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                    xmin, ymin, xmax, ymax = MpDetect.getRangeCoordinate(res.pose_landmarks, img_width=w, img_height=h)
                    pt1 = (int(xmin) - 20, int(ymin) - 20)
                    pt2 = (int(xmax) + 20, int(ymax) + 20)
                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
                cv2.imshow('frame', frame)
                if imageOutput:
                    cv2.imwrite(f"{images_file}/{count}.jpg", frame)  # 收集图片
                count += 1
                key = cv2.waitKey(1)
                if key & 0xff == ord('q'):
                    break
            else:
                print(f'cap.isOpened?{cap.isOpened()}')
                break
        cap.release()
        cv2.destroyAllWindows()

    ##############################################################################################
    # 以下是输入视频后的处理函数,还未测试
    ##############################################################################################
    def detectInputVideo(self, video_file, output_file, fps=20, fourcc='XVID'):
        count = 0
        cap = cv2.VideoCapture(video_file)
        pose = self.mpPose.Pose()
        while cap.isOpened():
            rvt, frame = cap.read()
            if not rvt:
                print("Can't receive frame.Exiting...")
                break
            h, w, c = frame.shape
            if not count:
                fourcc_code = cv2.VideoWriter_fourcc(*'{}'.format(fourcc))
                out = cv2.VideoWriter(output_file, fourcc_code, fps, (w, h))
                count += 1
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = pose.process(frameRGB)
            if res.pose_landmarks:
                self.mpDraw.draw_landmarks(frame, res.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                out.write(frame)
                cv2.imshow('frame', frame)
                key = cv2.waitKey(1)
                if key & 0xff == ord('q'):
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    ##############################################################################################

    ##############################################################################################
    # 获取所有节点的坐标值
    # 参数说明:
    # pose:mp.solutions.pose
    # resDotpose_landmarks:mp.solutions.pose.Pose.process().pose_landmarks,即上述的res.pose_landmarks
    # img_width:输入图片/视频帧的宽度
    # img_height:输入图片/视频帧的长度,如果是要返回所有节点的位置数据的话,则可以不用设置img_width和img_height
    # index:骨骼节点(int)
    # 示例代码:MpDetect.getCoordinates(self.mpPose, res.pose_landmarks, img_width=w, img_height=h)
    ##############################################################################################

    @staticmethod
    def getCoordinates(resDotpose_landmarks, img_width=1, img_height=1, index=-1, default=0, txtOut=False):
        nc = 0
        ##########################################################################################
        # 测试代码
        # print(
        #     f'Nose coordinates: ('
        #     f'{landmarks.landmark[pose.PoseLandmark.NOSE].x * img_width}, '
        #     f'{landmarks.landmark[pose.PoseLandmark.NOSE].y * img_height})'
        # )
        ##########################################################################################
        if not resDotpose_landmarks:
            print("没有找到节点")
        if 0 <= int(index) <= 32:
            coordinatesStr = '{}'.format(resDotpose_landmarks).split('landmark')[index + 1]
            coordinates = re.findall(r"\d+\.?\d*", coordinatesStr)  # 4个数值分别为:x,y,z,visibility
            x_coordinate = float(coordinates[0]) * img_width
            y_coordinate = float(coordinates[1]) * img_height
            return x_coordinate, y_coordinate
        # 所有节点的信息
        if int(index) == -1:
            bone_node_dict = {}  # 人体骨骼节点
            if txtOut:
                f = open(labels_file + "/" + str(default) + '.txt', 'a+')
            for i in range(33):
                coordinatesStr = '{}'.format(resDotpose_landmarks).split('landmark')[i + 1]
                coordinates = re.findall(r"\d+\.?\d*", coordinatesStr)
                xr = float(coordinates[0])  # x, y, z, visibility 数值为0~1
                yr = float(coordinates[1])
                zr = float(coordinates[2])
                vr = float(coordinates[3])
                bone_node_dict[inode_dict[i]] = (xr, yr, zr, vr,)
                if 11 <= i <= 14 or 23 <= i <= 24:
                    print("{0} {1} {2} {3} {4}".format(str(nc), xr, yr, ), file=f)
                    nc += 1
            if txtOut:
                f.close()
            return bone_node_dict
        else:
            print('index超出范围:[0~32]')

    # TODO 获取x和y轴方向上的最大值和最小值,算法后续再做改进    2022/7/17
    @staticmethod
    def getRangeCoordinate(resDotpose_landmarks, img_width, img_height):
        global now_ymax, now_xmax, now_ymin, now_xmin
        init_xmin, init_ymin = MpDetect.getCoordinates(resDotpose_landmarks, img_width, img_height, index=0)
        init_xmax = init_xmin
        init_ymax = init_ymin
        for i in range(1, 33):
            now_x, now_y = MpDetect.getCoordinates(resDotpose_landmarks, img_width, img_height, index=i)

            # 找最小位置点
            if now_x <= init_xmin:
                now_xmin = now_x
                init_xmin = now_xmin
            else:
                now_xmin = init_xmin

            if now_y <= init_ymin:
                now_ymin = now_y
                init_ymin = now_ymin
            else:
                now_ymin = init_ymin

            # 找最大位置点
            if now_x >= init_xmax:
                now_xmax = now_x
                init_xmax = now_xmax
            else:
                now_xmax = init_xmax

            if now_y >= init_ymax:
                now_ymax = now_y
                init_ymax = now_ymax
            else:
                now_ymax = init_ymax

        return now_xmin, now_ymin, now_xmax, now_ymax

    # 计算关键肢体间的角度
    def calAngle(self, node1, node2):
        pass
