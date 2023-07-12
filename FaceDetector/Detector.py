from abc import ABC, abstractmethod
import numpy as np
class Detector(ABC):

    def __init__(self,detectorConfig) -> None:
        self.config = detectorConfig
        self.results = []
    
    @abstractmethod
    def runInference(self,image):
        pass

    @abstractmethod
    def getResults(self):
        pass


class DetectionResults:
    def __init__(self,tlx,tly,brx,bry,classid,prob,meta=None) -> None:
        self.tl = (tlx,tly)
        self.br = (brx,bry)
        self.center = ((tlx + brx)/2,(tly + bry)/2)
        self.classid = classid
        self.prob = prob
        self.meta = meta
        self.center = np.asarray([(tlx + brx)/2,(tly + bry)/2])

    def getbbox(self):
        return (self.tl,self.br)
    
    def get_center(self):
        return self.center
        
    def getboxsize(self):
        return (self.br[0] - self.tl[0],self.br[1]-self.tl[1])
    
    def getboxCenter(self):
        x = int(self.tl[0] + (self.br[0]-self.tl[0])/2)
        y = int(self.tl[1] + (self.br[1]-self.tl[1])/2)
        return (x,y)
    
    def getMetadata(self,key=None):
        if key is None:
            return self.meta
        else:
            return self.meta[key]
        return None
        