from .adaFace import *
import torch,cv2
import numpy as np
from torchvision import transforms

class AdaFeatures:
    def __init__(self,modelArch,modelweights) -> None:
        self.model = build_model(modelArch)
        state_dict = torch.load(modelweights+'adaFace_ms1mv2.pt')['state_dict']
        model_statedict = {key[6:]:val for key,val in state_dict.items() if key.startswith('model.')}
        self.model.load_state_dict(model_statedict)
        self.model.eval()

    def runInference(self,image):
        img = ((image/255.)-0.5) /0.5
        imgTensor = torch.tensor(np.array([img.transpose(2,0,1)])).float()
        features,_ = self.model(imgTensor)
        return features