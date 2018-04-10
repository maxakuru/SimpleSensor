

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

        # Configs
        #self.moduleConfig = camConfigLoader.load(self.loggingQueue) #Get the config for this module
        self.config = baseConfig

        # Constants
        self._captureWidth = 1600
        self._captureHeight = 900
        self._numLEDS = 60
        self._collectionPointId = "tvcam1"
        self._collectionPointType = "ambiLED"

        # Logger
        self.logger = ThreadsafeLogger(loggingQueue, __name__)

    def run(self):
        """ Main thread method, run when the thread's start() function is called.
        Controls flow of detected faces and the MultiTracker. 
        Determines when to send 'reset' events to clients and when to send 'found' events. 
        This function contains various comments along the way to help understand the flow.
        You can use this flow, extend it, or build your own.
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

        while self.alive:
            ok, frame = self.video.read()
            if not ok:
                self.logger.error('Error while reading frame')
                break
                


    def initializeCamera(self):
        # open first webcam available
        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            self.video.open()

        #set the resolution from config
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, self._captureWidth)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, self._captureHeight)

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
        cv2.destroyAllWindows()
        # self.threadProcessQueue.join()
        time.sleep(1)
        self.exit = True

    #Custom methods for demo
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
