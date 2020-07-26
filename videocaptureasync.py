# file: videocaptureasync.py
import threading
import cv2
import copy
class VideoCaptureAsync:
    def __init__(self, width=2688, height=1520):
        # self.src = "rtsp://admin:DocoutBolivia@192.168.1.64:554/Streaming/Channels/102/"
        #self.src = '/home/docout/Desktop/Exportación de ACC - 2019-07-10 00.43.27.avi'
        
        #back office
        #self.src = 'rtsp://admin:S1stemas@172.16.20.95/onvif/profile1/media.smp'
        
        #boarding gate
        self.src = '/home/docout/Downloads/Friends Final scene HD.mp4'
        
        #avigilon sobre puerta
        #self.src = 'rtsp://administrator:@172.16.20.98/defaultPrimary?streamType=u'
        
        #self.src = '/home/docout/Desktop/Exportación de ACC - 2019-07-09 23.05.46.avi'
        self.src = "/home/docout/Desktop/snabox/motion-heatmap-opencv/input.mp4"

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
