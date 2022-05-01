from collections import deque
import cv2
from PIL import Image, ImageTk
import time
import tkinter as tk
from enum import Enum, unique
import numpy as np
import shapely
from shapely.geometry import LineString, Point

root = tk.Tk() 
root.geometry("1000x600")
    
scale_widget1 = tk.Scale(root, label = 'Canny threshold 1', from_=0, to=255, orient=tk.HORIZONTAL)
scale_widget1.set(45)
scale_widget1.place(x=10, y=0)


scale_widget2 = tk.Scale(root, label = 'Canny threshold 2', from_=0, to=255, orient=tk.HORIZONTAL)
scale_widget2.set(60)
scale_widget2.place(x=10, y=70)

scale_widget3 = tk.Scale(root, label = 'Area', from_=0, to=30000, orient=tk.HORIZONTAL)
scale_widget3.set(5500)
scale_widget3.place(x=10, y=140)

scale_minR = tk.Scale(root, label = 'Minimal diameter', from_=0, to=200, orient=tk.HORIZONTAL)
scale_minR.set(40)
scale_minR.place(x=10, y=280)

scale_maxR = tk.Scale(root, label = 'Maximal diameter', from_=0, to=300, orient=tk.HORIZONTAL)
scale_maxR.set(70)
scale_maxR.place(x=10, y=350)

scale_sigma = tk.Scale(root, label = 'sigma', from_=0, to=1, orient=tk.HORIZONTAL)
scale_sigma.set(0.33)
scale_sigma.place(x=10, y=420)

showCanny = tk.IntVar()
cannyCheckBox = tk.Checkbutton(root, text='Show Canny image',variable=showCanny, onvalue=1, offvalue=0)
cannyCheckBox.place(x=10, y=210)

def auto_canny(image, sigma=0.33):
	v = np.median(image)
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))

	return lower, upper

class Shape(Enum): 
    TRIANGLE = 3
    SQUARE = 4
    CIRCELE = 8
    PENTAGON = 5
    HEXAGON = 6
    
def printShape(img, approx, square_rect_param, intersections) :
    x, y, w, h = cv2.boundingRect(approx)
    position = (x, y)
    diff_geometrics = abs(w-h)
    
    if len(approx) == Shape.TRIANGLE.value:
        if(compare_shape_coordinates(approx, intersections, 20, 3)):
            cv2.putText(img, 'Trojuholnik', position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
  
    elif len(approx) == Shape.SQUARE.value:
        if(compare_shape_coordinates(approx, intersections, 20, 4)):
            txt = 'Obdlznik'
            if diff_geometrics < square_rect_param:
                txt = 'Stvorec'
            cv2.putText(img, txt, position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
  

def getLinesFromAcc(accumulator):
    result = []
    for line in accumulator:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        result.append([(x1,y1),(x2,y2)])
    return result

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y



def point_in_range(point1, point2, delta):
    result = False
    diff_x = abs(point1[0][0] - point2[0])
    diff_y = abs(point1[0][1] - point2[1])
    if(diff_x < delta and diff_y < delta):
        result = True
    return result 
def get_edges_approx(approx_edge_coordinates):
    edges = []
    for par in approx_edge_coordinates:
        edges.append([par[0][0], par[0][1]])
    return edges

def compare_shape_coordinates(shape1, shapes2, delta, ref_num_of_edges):
    result = False
    counter = 0
    for point1 in shape1:
        for point2 in shapes2:
            if point_in_range(point1, point2, delta):
                counter += 1
            break
    similar_point_diff = abs(counter - len(shape1))
    if(similar_point_diff >= ref_num_of_edges):
        result = True
    return result


def get_all_lines_intersections(lines):
    result = []
    for actual_line in lines:
        for next_line in lines:
            intersection = line_intersection(actual_line, next_line)
            if intersection:
                result.append(intersection)
    return result

def getContorous(img, imgContour, intersections):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > scale_widget3.get(): #Filter na velkost   
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            epsilon = 0.04*peri
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x+w, y+h), (0, 255, 0), 5)
            
            # cv2.putText(imgContour, "Points: " + str(len(approx)), (x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            # cv2.putText(imgContour, "Area: " + str(int(area)), (x+w+20, y+45), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
            printShape(imgContour, approx, 10, intersections)

def quit_(root):
    root.destroy()

def update_image(image_label, cam, image_canny_label):
    (readsuccessful, f) = cam.read()
    rgb_img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        
    imgBlur = cv2.GaussianBlur(hsv_img, (7, 7), 1)
    h, s, v = cv2.split(imgBlur)
    # imgGary = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    imgSat = s
    # cannyLow, cannyUp = auto_canny(rgb_img)
    
    otsuThreshold,th3 = cv2.threshold(imgSat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cannyUp = otsuThreshold
    cannyLow = 0.5 * otsuThreshold
    scale_widget1.set(cannyLow)
    scale_widget2.set(cannyUp)
    imgCanny = cv2.Canny(imgSat, scale_widget1.get(), scale_widget2.get(),apertureSize = 3)
    
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations = 1)
    
    linesAcccumulator = cv2.HoughLines(imgCanny, 1, np.pi/180,100)
    lines = getLinesFromAcc(linesAcccumulator)
    intersections = get_all_lines_intersections(lines)
    
    getContorous(imgDil, rgb_img, intersections)
  
    circles = cv2.HoughCircles(imgSat, cv2.HOUGH_GRADIENT, 1, minDist=150, param1=cannyUp, param2=30, minRadius=scale_minR.get(), maxRadius=scale_maxR.get())

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x,y,r) in circles:
            cv2.circle(rgb_img, (x,y), r, (36,255,12), 3)
            cv2.putText(rgb_img, 'Kruh', (x,y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # print("Img canny" + str(imgCanny.shape))
    dsize = (300, 250)
    imgCanny = cv2.resize(imgCanny, dsize)
    cannyA = Image.fromarray(imgCanny)
    cannyB = ImageTk.PhotoImage(image=cannyA)
    image_canny_label.configure(image=cannyB)
    image_canny_label._image_cache = cannyB
    
    if showCanny.get():
        image_canny_label.place(relx = 1, rely = 0, anchor="ne")
    else:
        image_canny_label.place_forget()
    image_canny_label.config(width=300, height=250)
    
    
    a = Image.fromarray(rgb_img)
    b = ImageTk.PhotoImage(image=a)
    image_label.configure(image=b)
    image_label._image_cache = b  # avoid garbage collection
    image_label.place(relx=0.5, rely=0.5, anchor='center')
    
    
    root.update()


def update_fps(fps_label):
    frame_times = fps_label._frame_times
    frame_times.rotate()
    frame_times[0] = time.time()
    sum_of_deltas = frame_times[0] - frame_times[-1]
    count_of_deltas = len(frame_times) - 1
    try:
        fps = int(float(count_of_deltas) / sum_of_deltas)
    except ZeroDivisionError:
        fps = 0
    fps_label.configure(text='FPS: {}'.format(fps))


def update_all(root, image_label, cam, fps_label, image_canny_label):
    update_image(image_label, cam, image_canny_label)
    update_fps(fps_label)
    root.after(20, func=lambda: update_all(root, image_label, cam, fps_label,image_canny_label))


if __name__ == '__main__':

    image_label = tk.Label(master=root)# label for the video frame
    image_canny_label = tk.Label(master=root)# label for the video frame
    image_label.pack()

    cam = cv2.VideoCapture(0) 
    fps_label = tk.Label(master=root)# label for fps
    fps_label._frame_times = deque([0]*5)  # arbitrary 5 frame average FPS
    fps_label.pack()
    # quit button
    quit_button = tk.Button(master=root, text='Quit',command=lambda: quit_(root))
    quit_button.place()
    quit_button.pack()
    # setup the update callback
    root.after(0, func=lambda: update_all(root, image_label, cam, fps_label, image_canny_label))
    root.mainloop()