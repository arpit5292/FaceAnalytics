# from .yolov8 import YOLO
from .mtcnn import MTCNN

from .Detector import Detector, DetectionResults
import torch

class FaceDetector(Detector):
    def __init__(self, detectorType) -> None:
        self.config = detectorType
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.__loadModel(detectorType)
        self.model.eval()
        self.results = []

    def __loadModel(self,modelType):
        if modelType == 'MTCNN':
            model = MTCNN(device=self.device)
        if modelType == 'YOLO':
            # self.model = YOLO(self.config['weights'])
            # self.model.to(device)
            pass
        return model
        
    def runInference(self, image):
        self.results.clear()
        if self.config == 'MTCNN':
            boxes,probs,landmarks = self.model.detect(image,landmarks=True)
            if boxes is None:
                return
            for box,conf,kp in zip(boxes,probs,landmarks):
                ob = DetectionResults(int(box[0]),int(box[1]),int(box[2]),int(box[3]),'face',conf,meta=kp)
                self.results.append(ob)
        

    def getResults(self):
        return self.results
        
