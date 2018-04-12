import cv2
from threading import Thread
import time
import numpy as np
from scipy.stats import itemfreq
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
        self._numLEDs = 60
        self._collectionPointId = "tvcam1"
        self._collectionPointType = "ambiLED"
        self._showVideoStream = True
        self._delimiter = ';'
        self._colorMode = 'edgeDominant'
        # self._colorMode = 'edgeMean'
        self._perimeterDepth = 20
        self._topSegments = 3
        self._sideSegments = 2

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
                if k == 32 and self.boundSet:
                    # on space, user wants to finalize bounds, only allow them to exit if bounds set
                    cont = False
                # elif k != 27:
                    # any other key clears rectangles
                    # framecopy = frame.copy()
                    #ok, frame = self.video.read()
                    # cv2.imshow('Set ROI', framecopy)
                    # cv2.setMouseCallback('Set ROI', self.getROI, framecopy)
        cv2.destroyWindow('Set ROI')

        self.initKMeans()

        # Set up for all modes
        top_length_pixels = self.fx-self.ix
        side_length_pixels = self.fy-self.iy
        perimeter_length_pixels = top_length_pixels*2 + side_length_pixels*2

        # mode specific setup
        if self._colorMode == 'dominant':
            pass
        if self._colorMode == 'edgeDominant' or self._colorMode == 'edgeMean':
            perimeter_depth = 0
            if self._perimeterDepth < side_length_pixels/2 and self._perimeterDepth < top_length_pixels/2:
                perimeter_depth = self._perimeterDepth
            else: 
                perimeter_depth = min(side_length_pixels/2, top_length_pixels/2)

        while self.alive:
            ok, ogframe = self.video.read()
            if not ok:
                self.logger.error('Error while reading frame')
                break
            frame = ogframe.copy()

            # Dominant color
            if self._colorMode == 'dominant':
                dominant_color = self.getDominantColor(cv2.resize(frame[:,:,:], (0,0), fx=0.4, fy=0.4), self.ix, self.fx, self.iy, self.fy)
                #self.putCPMessage(dominant_color, 'light-dominant')
                print(dominant_color)

            elif self._colorMode == 'edgeMean':
                data = self.getEdgeMeanColors(frame, top_length_pixels, side_length_pixels, perimeter_length_pixels, perimeter_depth)

            elif self._colorMode == 'edgeDominant':
                # this is the most promising
                self.getEdgeDominantColors(frame, top_length_pixels, side_length_pixels, perimeter_length_pixels, perimeter_depth)

            if self._showVideoStream:
                cv2.rectangle(frame, (self.ix, self.iy), (self.fx, self.fy), (255,0,0), 1)
                cv2.imshow("output", frame)
                cv2.waitKey(1)

    def getMeanColor(self, frame):
        color = [frame[:,:,i].mean() for i in range(frame.shape[-1])]
        return color

    def initKMeans(self):
         # kmeans vars
        self.n_colors = 5
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        self.flags = cv2.KMEANS_RANDOM_CENTERS

    def getDominantSegmentColor(self, segment):
        average_color = [segment[:,:,i].mean() for i in range(segment.shape[-1])]
        arr = np.float32(segment)
        pixels = arr.reshape((-1, 3))

        # kmeans clustering
        _, labels, centroids = cv2.kmeans(pixels, self.n_colors, None, self.criteria, 10, self.flags)

        palette = np.uint8(centroids)
        quantized = palette[labels.flatten()]
        quantized = quantized.reshape(segment.shape)

        dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]

        return dominant_color

    def getEdgeMeanColors(self, frame, top_length_pixels, side_length_pixels, perimeter_length_pixels, perimeter_depth):
        # assuming LEDs are evenly distributed, find number for each edge of ROI
        top_num_leds = self._numLEDs*(top_length_pixels/perimeter_length_pixels)
        side_num_leds = self._numLEDs*(side_length_pixels/perimeter_length_pixels)
        top_segment_length = top_length_pixels/self._topSegments
        side_segment_length = side_length_pixels/self._sideSegments

        for i in range(0, self._topSegments):
            ix = int(self.ix+i*top_segment_length)
            fx = int(self.ix+(i+1)*top_segment_length)
            iy = int(self.iy)
            fy = int(self.iy+perimeter_depth)
            c = self.getMeanColor(cv2.resize(frame[ix:fx, iy:fy, :], (0,0), fx=0.2, fy=0.2))
            print('top segment %s, c: '%i, c)
            if self._showVideoStream:
                cv2.rectangle(frame, (ix, iy), (fx, fy), (0,0,255), 1)
                cv2.rectangle(frame, (ix, iy-(10+perimeter_depth)), (fx, fy-perimeter_depth), (int(c[0]), int(c[1]), int(c[2])), 10)

        for i in range(0, self._sideSegments):
            ix = int(self.fx-perimeter_depth)
            fx = int(self.fx)
            iy = int(self.iy+i*side_segment_length)
            fy = int(self.iy+(i+1)*side_segment_length)
            c = self.getMeanColor(cv2.resize(frame[ix:fx, iy:fy, :], (0,0), fx=0.2, fy=0.2))
            print('right segment %s, c: '%i, c)
            if self._showVideoStream:
                cv2.rectangle(frame, (ix, iy), (fx, fy), (0,255,0), 1)
                cv2.rectangle(frame, (ix+perimeter_depth, iy), (fx+(10+perimeter_depth), fy), (int(c[0]), int(c[1]), int(c[2])), 10)

        for i in range(0, self._topSegments):
            ix = int(self.fx-(i+1)*top_segment_length) 
            fx = int(self.fx-i*top_segment_length)
            iy = int(self.fy-perimeter_depth)
            fy = int(self.fy)
            c = self.getMeanColor(cv2.resize(frame[ix:fx, iy:fy, :], (0,0), fx=0.2, fy=0.2))
            print('bottom segment %s, c: '%i, c)
            if self._showVideoStream:
                cv2.rectangle(frame, (ix, iy), (fx, fy), (0,0,255), 1)
                cv2.rectangle(frame, (ix, iy+perimeter_depth), (fx, fy+(10+perimeter_depth)), (int(c[0]), int(c[1]), int(c[2])), 10)

        for i in range(0, self._sideSegments):
            ix = int(self.ix)
            fx = int(self.ix+perimeter_depth)
            iy = int(self.fy-(i+1)*side_segment_length)
            fy = int(self.fy-i*side_segment_length)
            c = self.getMeanColor(cv2.resize(frame[ix:fx, iy:fy, :], (0,0), fx=0.2, fy=0.2))
            print('left segment %s, c: '%i, c)
            if self._showVideoStream:
                cv2.rectangle(frame, (ix, iy), (fx, fy), (0,255,0), 1)
                cv2.rectangle(frame, (ix-(10+perimeter_depth), iy), (fx-perimeter_depth, fy), (int(c[0]), int(c[1]), int(c[2])), 10)

        return 0

    def getEdgeDominantColors(self, frame, top_length_pixels, side_length_pixels, perimeter_length_pixels, perimeter_depth):
        # assuming LEDs are evenly distributed, find number for each edge of ROI
        top_num_leds = self._numLEDs*(top_length_pixels/perimeter_length_pixels)
        side_num_leds = self._numLEDs*(side_length_pixels/perimeter_length_pixels)
        top_segment_length = top_length_pixels/self._topSegments
        side_segment_length = side_length_pixels/self._sideSegments

        for i in range(0, self._topSegments):
            ix = int(self.ix+i*top_segment_length)
            fx = int(self.ix+(i+1)*top_segment_length)
            iy = int(self.iy)
            fy = int(self.iy+perimeter_depth)
            c = self.getDominantSegmentColor(cv2.resize(frame[ix:fx, iy:fy, :], (0,0), fx=0.2, fy=0.2))
            if self._showVideoStream:
                cv2.rectangle(frame, (ix, iy), (fx, fy), (0,0,255), 1)
                cv2.rectangle(frame, (ix, iy-(10+perimeter_depth)), (fx, fy-perimeter_depth), (int(c[0]), int(c[1]), int(c[2])), 10)

        for i in range(0, self._sideSegments):
            ix = int(self.fx-perimeter_depth)
            fx = int(self.fx)
            iy = int(self.iy+i*side_segment_length)
            fy = int(self.iy+(i+1)*side_segment_length)
            ff = cv2.resize(frame[ix:fx, iy:fy, :], (0,0), fx=0.2, fy=0.2)
            c = self.getDominantSegmentColor(ff)
            if self._showVideoStream:
                cv2.rectangle(frame, (ix, iy), (fx, fy), (0,255,0), 1)
                cv2.rectangle(frame, (ix+perimeter_depth, iy), (fx+(10+perimeter_depth), fy), (int(c[0]), int(c[1]), int(c[2])), 10)

        for i in range(0, self._topSegments):
            ix = int(self.fx-(i+1)*top_segment_length) 
            fx = int(self.fx-i*top_segment_length)
            iy = int(self.fy-perimeter_depth)
            fy = int(self.fy)
            c = self.getDominantSegmentColor(cv2.resize(frame[ix:fx, iy:fy, :], (0,0), fx=0.2, fy=0.2))
            if self._showVideoStream:
                cv2.rectangle(frame, (ix, iy), (fx, fy), (0,0,255), 1)
                cv2.rectangle(frame, (ix, iy+perimeter_depth), (fx, fy+(10+perimeter_depth)), (int(c[0]), int(c[1]), int(c[2])), 10)

        for i in range(0, self._sideSegments):
            ix = int(self.ix)
            fx = int(self.ix+perimeter_depth)
            iy = int(self.fy-(i+1)*side_segment_length)
            fy = int(self.fy-i*side_segment_length)
            c = self.getDominantSegmentColor(cv2.resize(frame[ix:fx, iy:fy, :], (0,0), fx=0.2, fy=0.2))
            if self._showVideoStream:
                cv2.rectangle(frame, (ix, iy), (fx, fy), (0,255,0), 1)
                cv2.rectangle(frame, (ix-(10+perimeter_depth), iy), (fx-perimeter_depth, fy), (int(c[0]), int(c[1]), int(c[2])), 10)

        return 0

    def getDominantColor(self, img, ix, fx, iy, fy):
        ix = int(ix)
        fx = int(fx)
        iy = int(iy)
        fy = int(fy)
        average_color = [img[ix:fx, iy:fy, i].mean() for i in range(img.shape[-1])]
        arr = np.float32(img)
        pixels = arr.reshape((-1, 3))

        # kmeans clustering
        _, labels, centroids = cv2.kmeans(pixels, self.n_colors, None, self.criteria, 10, self.flags)

        palette = np.uint8(centroids)
        quantized = palette[labels.flatten()]
        quantized = quantized.reshape(img.shape)

        dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]

        return dominant_color

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

        elif type == "light-edges":
            # Reset collection start and now needs needs reset
            collectionStart = time.time()

            self.logger.info('Sending light message')
            msg = CollectionPointEvent(
                self._collectionPointId,
                self._collectionPointType,
                'light-edges',
                data
            )
            self.outQueue.put(msg)

        elif type == "light-dominant":
            # Reset collection start and now needs needs reset
            collectionStart = time.time()

            self.logger.info('Sending light message')
            msg = CollectionPointEvent(
                self._collectionPointId,
                self._collectionPointType,
                'light-dominant',
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
