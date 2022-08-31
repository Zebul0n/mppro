from mpdetect import MpDetect

md = MpDetect()
img_dir = r"D:\1python_java_C\openposeProject\mediapipeStudy\mpPro\images"
output_dir = r"D:\1python_java_C\openposeProject\mediapipeStudy\mpPro\run"
# var = md.detectImg(img_dir, output_dir)
# print(var)

md.detectCam(0, imageOutput=True)
