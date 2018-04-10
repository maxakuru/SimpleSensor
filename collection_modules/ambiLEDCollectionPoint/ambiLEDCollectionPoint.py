import cv2
from threading import Thread
import time
from collectionPointEvent import CollectionPointEvent
from threadsafeLogger import ThreadsafeLogger

class TVCollectionPoint(Thread):

    def __init__(self, baseConfig, pInBoundQueue, pOutBoundQueue, loggingQueue):
        """ Initialize new TVCollectionPoint instance.
        Setup queues, variables, configs, constants and loggers.
        """

        super(TVCollectionPoint, self).__init__()

        if not self.check_opencv_version("3.", cv2):
            print("OpenCV version {0} is not supported. Use 3.x for best results.".format(self.get_opencv_version()))

        # Queues
        self.outQueue = pOutBoundQueue #messages from this thread to the main process
        self.inQueue= pInBoundQueue
        self.loggingQueue = loggingQueue

        # Variables
        self.video = None
        self.alive = True
        self.ix = -1
        self.iy = -1
        self.fx = -1
        self.fy = -1
        self.clicking = False
        self.boundSet = False

        self.x1,self.x2,self.y1,self.y2 = 0,0,0,0

        # Configs
        #self.moduleConfig = camConfigLoader.load(self.loggingQueue) #Get the config for this module
        self.config = baseConfig

        # Constants
        self._captureWidth = 1600
        self._captureHeight = 900
        self._numLEDS = 60
        self._collectionPointId = "tvcam1"
        self._collectionPointType = "ambiLED"
        self._showVideoStream = True
        self._delimiter = ';'

        # Logger
        self.logger = ThreadsafeLogger(loggingQueue, __name__)

    def run(self):
        """ Main thread method, run when the thread's start() function is called.
        Controls flow of detected faces and the MultiTracker. 
        Sends color data in string format, like "#fffff;#f1f1f1;..."
        """

        # Monitor inbound queue on own thread
        self.threadProcessQueue = Thread(target=self.processQueue)
        self.threadProcessQueue.setDaemon(True)
        self.threadProcessQueue.start()

        self.initializeCamera()

        # Setup timer for FPS calculations
        start = time.time()
        frameCounter = 1
        fps = 0

        # Start timer for collection events
        self.collectionStart = time.time()

        ok, frame = self.video.read()
        if not ok:
            self.logger.error('Cannot read video file')
            self.shutdown()
        else:
            framecopy = frame.copy()
            cont = True
            while cont or not self.boundSet:
                cv2.imshow('Set ROI', framecopy)
                cv2.setMouseCallback('Set ROI', self.getROI, frame)
                k = cv2.waitKey(0)
                if k == 27 and self.boundSet:
                    # on esc, user wants to exit, only allow them to exit if bounds set
                    print('k: ', k)
                    cont = False
                # elif k != 27:
                    # any other key clears rectangles
                    # framecopy = frame.copy()
                    #ok, frame = self.video.read()
                    # cv2.imshow('Set ROI', framecopy)
                    # cv2.setMouseCallback('Set ROI', self.getROI, framecopy)
        cv2.destroyWindow('Set ROI')


        while self.alive:
            ok, frame = self.video.read()
            if not ok:
                self.logger.error('Error while reading frame')
                break
                
            if self._showVideoStream:
                cv2.rectangle(frame, (self.ix, self.iy), (self.fx, self.fy), (255,0,0), 1)
                cv2.imshow("output", frame)
                cv2.waitKey(1)


    def initializeCamera(self):
        # open first webcam available
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            self.video.open()

        #set the resolution from config
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self._captureWidth)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self._captureHeight)

    def getROI(self, event, x, y, flags, frame):
        framecopy = frame.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicking = True
            self.ix,self.iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.clicking:
                cv2.rectangle(framecopy, (self.ix,self.iy),(x,y),(0,255,0),-1)
                cv2.imshow('Set ROI', framecopy)

        elif event == cv2.EVENT_LBUTTONUP:
            self.clicking = False
            cv2.rectangle(framecopy, (self.ix,self.iy),(x,y),(0,255,0),-1)
            cv2.imshow('Set ROI', framecopy)
            self.fx,self.fy = x,y
            self.boundSet = True

    def processQueue(self):
        self.logger.info("Starting to watch collection point inbound message queue")
        while self.alive:
            if (self.inQueue.empty() == False):
                self.logger.info("Queue size is %s" % self.inQueue.qsize())
                try:
                    message = self.inQueue.get(block=False,timeout=1)
                    if message is not None:
                        if message == "SHUTDOWN":
                            self.logger.info("SHUTDOWN command handled on %s" % __name__)
                            self.shutdown()
                        else:
                            self.handleMessage(message)
                except Exception as e:
                    self.logger.error("Unable to read queue, error: %s " %e)
                    self.shutdown()
                self.logger.info("Queue size is %s after" % self.inQueue.qsize())
            else:
                time.sleep(.25)

    def handleMessage(self, message):
        self.logger.info("handleMessage not implemented!")    
       
    def putCPMessage(self, data, type):
        if type == "off":
            # Send off message
            self.logger.info('Sending off message')
            msg = CollectionPointEvent(
                self._collectionPointId,
                self._collectionPointType,
                'off',
                None)
            self.outQueue.put(msg)

        elif type == "light":
            # Reset collection start and now needs needs reset
            collectionStart = time.time()

            self.logger.info('Sending light message')
            msg = CollectionPointEvent(
                self._collectionPointId,
                self._collectionPointType,
                'light',
                data
            )
            self.outQueue.put(msg)

    def shutdown(self):
        self.alive = False
        self.logger.info("Shutting down")
        # self.putCPMessage(None, 'off')
        cv2.destroyAllWindows()
        time.sleep(1)
        self.exit = True

    def get_opencv_version(self):
        import cv2 as lib
        return lib.__version__

    def check_opencv_version(self,major, lib=None):
        # if the supplied library is None, import OpenCV
        if lib is None:
            import cv2 as lib

        # return whether or not the current OpenCV version matches the
        # major version number
        return lib.__version__.startswith(major)
