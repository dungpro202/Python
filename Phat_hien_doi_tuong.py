from PIL import ImageFont, ImageDraw, Image
from cv2 import cv2
import numpy as np

from tkinter import filedialog
from tkinter import *
# Load Yolo
# Đưa dữ liệu vào mạng noron dnn
# tên các class đưa vào mảng classes[]
net = cv2.dnn.readNet("yolov3_custom_last4000.weights", "obj.cfg")
classes = []
with open("obj.name", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# randonm color
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#Lấy hình ảnh vào
from tkinter import filedialog
from tkinter import *
root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
print (root.filename)

# Loading image
img = cv2.imread(root.filename)
img = cv2.resize(img,None,fx=1,fy=1) 

#img = cv2.resize(img,(800,800))
height, width, channels = img.shape

# Đưa hình ảnh vào mạng để phát hiện đồ vật, hình ảnh được resize lại kích thước 416x416  
# tỷ lệ 1/255 pixel = 0.00392 vì trong yolov3 hình ảnh đc chuẩn hóa thành 0->1
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

#đưa hình ảnh vào mạng
net.setInput(blob)
outs = net.forward(output_layers)

# Hiển thị thông tin trên màn hình
class_ids = []
confidences = []
boxes = []

for out in outs:

    # pc= xác suất xuất hiện của đối tượng trong box , sau đó nhân với vectơ class. giá trị nào lớn nhất chính là score
    # lấy score cao nhất của điểm được phát hiện
    # suy ra class_id từ scores ở vị trí của nó
    # độ tự tin là điểm cao nhất đó
    for detection in out:
        scores = detection[5:] 
        class_id = np.argmax(scores) #suy ra class_id từ scores ở vị trí có giá trị lớn nhất
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Rectangle cordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            #print('aaaaa')
            #print(detection[0:18])
           
 
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
#conf_threshold = 0.5
#nms_threshold = 0.4
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#Vẽ indexes 
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])+"-"+str(round(confidences[i],3))
        
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)
        
        print(label)
       



cv2.imshow("Nhan_dien_trang_phuc",img)
cv2.waitKey(0)
cv2.destroyAllWindows()