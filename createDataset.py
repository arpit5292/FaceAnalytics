from FaceFeatures import AdaFeatures
from FaceDetector import FaceDetector
from FaceDetector.alignFace import get_reference_facial_points,warp_and_crop_face

import os,cv2,joblib,torch

class FaceRecognitionDataSet:
    def __init__(self,dataset_path,detector=None,featureNet=None) -> None:
        self.basePath = dataset_path
        self.embedingsFile = os.path.join(dataset_path,'dataset.joblib')
        if detector is None:
            self.detector = FaceDetector("MTCNN")
        else:
            self.detector = detector
        if featureNet is None:
            self.featurenet = AdaFeatures('ir_50','./data/weights/')
        else:
            self.featurenet = featureNet
        self.refPoint = get_reference_facial_points(default_square=True)

        if not os.path.isfile(self.embedingsFile):
            self.createDataset(dataset_path)

        
        
    def createDataset(self,dataset_path):
        print(f'Creating dataset from {dataset_path}')
        identity = os.listdir(dataset_path)
        if len(identity)==0:
            raise f"dataset {dataset_path} should have atleast 1 person folder and images"

        embeddings = {}
        for id in identity:
            print(f'Extracting {id} features')
            imgFiles = os.listdir(os.path.join(dataset_path,id))
            if len(imgFiles) ==0:
                print(f"{id} must have atleast 1 image in folder")
                continue
            idFeatures = []
            for img in imgFiles:
                p = os.path.join(dataset_path,id,img)
                img = cv2.imread(p,cv2.IMREAD_COLOR)
                self.detector.runInference(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                results = self.detector.getResults()
                print(len(results))
                if len(results)>1 or len(results)==0:
                    continue
                kp = results[0].getMetadata()
                wrapedFace = warp_and_crop_face(img,kp,self.refPoint,(112,112))
                embds = self.featurenet.runInference(wrapedFace)
                idFeatures.append(embds.cpu().detach().numpy())
            if len(idFeatures) == 0:
                print(f"Can not find face for {id}")
                return
            embeddings[id] = idFeatures
        joblib.dump(embeddings,self.embedingsFile)

    def getDatasetEmbeddings(self):
        embeddings = joblib.load(self.embedingsFile)
        ids = list(embeddings.keys())
        vect = [ torch.from_numpy(embeddings[x][0]) for x in embeddings]
        vect = torch.cat(vect)
        return vect,ids
            

                
                    


        
