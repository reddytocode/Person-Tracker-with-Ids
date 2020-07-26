#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from videocaptureasync import *

cv2.namedWindow("window", cv2.WINDOW_AUTOSIZE)
def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    #async
    #| = cv2.VideoCapture("rtsp://admin:DocoutBolivia@192.168.1.64:554/Streaming/Channels/102/")
    video_capture = VideoCaptureAsync()
    #video_capture = cv2.VideoCapture("/home/docout/Desktop/Exportación de ACC - 2019-07-10 00.43.27.avi")
    #video_capture = cv2.VideoCapture('rtsp://admin:S1stemas@172.16.20.116/onvif/profile1/media.smp')
    #video_capture = cv2.VideoCapture('rtsp://admin:S1stemas@172.16.20.95/onvif/profile1/media.smp')
    
    #video_capture = cv2.VideoCapture("/home/docout/Desktop/spot_telf_exported.mp4")
    #video_capture = cv2.VideoCapture("/home/docout/Desktop/atb.mp4")

    
    video_capture.start()
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        #w = int(video_capture.get(3))
        #h = int(video_capture.get(4))

        w = 2688
        h = 1520
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1 
        
    fps = 0.0
    contFrame = 0
    cont_person_passs = 0
    v_people = [True]

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        ret, frame = video_capture.read()



        if(contFrame == 1):
            if ret != True:
                break
            t1 = time.time()


        #plot the image
            #frame = frame[300:1080, 600:1580]
            #plt.imshow(frame)
            #plt.show()

        # image = Image.fromarray(frame)
            cv2.imshow("org", frame)
            frame = frame [300:1080, 600:1580]
            image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            boxs = yolo.detect_image(image)
        # print("box_num",len(boxs))
            features = encoder(frame,boxs)
            
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            cv2.line(frame, (0, 300), (900, 300), (255, 0, 0), 2)
            cv2.line(frame, (0, 500), (900, 500), (255, 0, 0), 2)
            cv2.putText(frame, str(cont_person_passs), (800, 650), 0, 5e-3 * 200, (0, 255, 0), 2)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                #track.
                #print(track.track_id, " pos y: ", int(bbox[1]))

                print(track.track_id)
                if (len(v_people) == track.track_id):
                    v_people.append(False)
                    print("added: len ", len(v_people))
                elif (len(v_people) < track.track_id):
                    while True:
                        if (len(v_people) == track.track_id):
                            break
                        else:
                            v_people.append(False)

                color_pass = (255, 255, 255)
                center = int(bbox[1]) + (int(bbox[3]) - int(bbox[1])) /2
                centerX = int(bbox[0]) + (int(bbox[2]) - int(bbox[0]))/2
                cv2.circle(frame, (int(centerX), int(bbox[1])), 5, (255, 210, 123), 2)
                #if ((int(bbox[1] < 500 and int(bbox[1] > 400))) or (int(bbox[3]) < 500 and int(bbox[3]) > 400)):
                if (int(bbox[1]) < 500 and int(bbox[1]) > 300 and track.age > 80):
                    print("passed")
                    cv2.line(frame, (0, 300), (900, 300), (0, 0, 255), 2)
                    cv2.line(frame, (0, 500), (900, 500), (0, 0, 255), 2)

                    #print(len(v_people), " and id: ", track.track_id)
                    if(v_people[track.track_id] == False):
                        color_pass = (0, 0, 255)
                        v_people[track.track_id] = True
                        cont_person_passs += 1

                    print(cont_person_passs)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),color_pass, 1)
                text_put = str(track.track_id) + " : "+ str(track.age)
                cv2.putText(frame, text_put,(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),1)
                if(track.track_id == 0):
                    print("\n\n\nprimero!!!!\n\n\n")

            """for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)"""

            #makes it full screen
            #cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
            #cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('window', frame)
            
            #if writeVideo_flag:
            #    # save a frame
            #    out.write(frame)
            #    frame_index = frame_index + 1
            #    list_file.write(str(frame_index)+' ')
            #    if len(boxs) != 0:
            #        for i in range(0,len(boxs)):
            #            list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            #    list_file.write('\n')
            #    
            #fps  = ( fps + (1./(time.time()-t1)) ) / 2
            #print("fps= %f"%(fps))
            
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            contFrame = 0
        
        contFrame += 1

    video_capture.stop()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())