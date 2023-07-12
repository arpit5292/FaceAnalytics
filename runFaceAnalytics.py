from argparse import ArgumentParser
import cv2,time
import json
import torch
import numpy as np
from sort import Sort
from FaceDetector import FaceDetector
from FaceDetector.alignFace import get_reference_facial_points,warp_and_crop_face
from createDataset import FaceRecognitionDataSet
from StreamProcessor import streamProcessor
from tracker import Tracker
from FaceFeatures import AdaFeatures

class VideoAnalytics:

    # def judge_side_face(self,facial_landmarks):      
    #   wide_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[1])
    #   high_dist = np.linalg.norm(facial_landmarks[0] - facial_landmarks[3])
    #   dist_rate = high_dist / wide_dist


    #   # cal std
    #   vec_A = facial_landmarks[0] - facial_landmarks[2]
    #   vec_B = facial_landmarks[1] - facial_landmarks[2]
    #   vec_C = facial_landmarks[3] - facial_landmarks[2]
    #   vec_D = facial_landmarks[4] - facial_landmarks[2]
    #   dist_A = np.linalg.norm(vec_A)
    #   dist_B = np.linalg.norm(vec_B)
    #   dist_C = np.linalg.norm(vec_C)
    #   dist_D = np.linalg.norm(vec_D)

    #   # cal rate
    #   high_rate = dist_A / dist_C
    #   if dist_C > dist_D:
    #       width_rate = dist_C / dist_D
    #   else:
    #       width_rate = dist_D / dist_C
    #   high_ratio_variance = np.fabs(high_rate - 1.1)  # smaller is better
    #   width_ratio_variance = np.fabs(width_rate - 1) 

    #   return dist_rate, high_ratio_variance, width_ratio_variance            


    def __init__(self,config,inputPath) -> None:
        self.analyticsConfig = config
        self.detectorPipeline = []
        self._createDetectionEngines()
        self.metaPipeline = []
        self._createMetaEngines()
        self.inputPath = inputPath
        self.stream = streamProcessor(inputPath,0)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.refPoint = get_reference_facial_points(default_square=True)
        data = FaceRecognitionDataSet(config['dataset'],self.detectorPipeline[0],self.metaPipeline[0])
        self.savedFeaturs,self.savedId = data.getDatasetEmbeddings()
        self.tracker = Tracker(100, 15, 80)

    def _createDetectionEngines(self):
        for x in self.analyticsConfig['detector']:
            self.detectorPipeline.append(FaceDetector(x['type']))
    
    def _createMetaEngines(self):
        for x in self.analyticsConfig['metadata']:
            self.metaPipeline.append(AdaFeatures('ir_50','./data/weights/'))

    def runDetectionPipeline(self):
        delay = 0
        print(f'Processing {self.inputPath}')
        colours = np.random.rand(32, 3)
        
        # cam = cv2.VideoCapture(video_name)
        frame_width = self.stream.get_3
        frame_height = self.stream.get_4
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        tracker = Sort()
        
        # ret = True
        out = cv2.VideoWriter( 'outputs1.mp4', fourcc, 20.0, (1280,720))         
        while 1:            
            img = self.stream.read()
            
            if img is None:
                break
            img_size = np.asarray(img.shape)[0:2]
            procImg = img.copy()
            inferimg = cv2.cvtColor(procImg,cv2.COLOR_BGR2RGB)
            tms = time.time()
            face_list = []  
            final_faces = []
            face_addtional_attribute = []                
            for x in self.detectorPipeline:
                x.runInference(inferimg)
                results = x.getResults()
                features = []
                centers = []  
        
                for subj in results:
                    
                    center = subj.get_center()  
                    centers.append(center)                        
                    bbox = subj.getbbox()
                    bw,bh = subj.getboxsize()                                      
                    wraped_face = warp_and_crop_face(procImg,subj.getMetadata(),self.refPoint,(112,112))
                    item = [bbox[0][0],bbox[0][1], bbox[1][0],bbox[1][1]]
                    face_list.append(item)
                    face_addtional_attribute.append(wraped_face)
                    # if self.analyticsConfig['mode'] == 'fr':
                        # subj.getMetadata()
                        # cv2.imshow('Face',wraped_face)
                        # for met in self.metaPipeline:
                        #     embd =met.runInference(wraped_face)
                        #     sim = self.savedFeaturs @ embd.T
                        #     idx = sim.argmax()
                        #     # print(sim.T.detach().numpy().tolist())
                        #     # print(self.savedId)
                        #     prob =sim[idx].detach().numpy()[0] 
                        #     subId = self.savedId[idx]
                        #     # features.append(embd)
                        #     print(prob,subId)
                        #     if prob<=0.20:
                        #         # cv2.putText(img,"{:.2f}".format(prob)+f'-{subId}',(bbox[0][0]+10,bbox[0][1]+20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)
                        #     if prob>0.20:
                                # cv2.putText(img,"{:.2f}".format(prob)+f'-{subId}',(bbox[0][0]+10,bbox[0][1]+20),cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,0),2)
                                # print(prob,subId)
                        # cv2.waitKey()
                    # cv2.rectangle(img,bbox[0],bbox[1],(0,0,255),4)
                    # cv2.putText(img,"{:.2f}".format(subj.prob),bbox[0],cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,0),2)
                    # cv2.putText(img,"{0}X{1}".format(bw,bh),bbox[1],cv2.FONT_HERSHEY_PLAIN,1.5,(255,255,0),2)
            if len(face_list) != 0:                   
                final_faces = np.array(face_list)               
            trackers,global_id_dict = tracker.update(final_faces,img_size,face_addtional_attribute,self.savedFeaturs,self.savedId)                                       
            for trk in trackers:
                d = trk[0:5].astype(np.int32)
                cv2.rectangle(img, (d[0], d[1]), (d[2], d[3]), colours[d[4] % 32, :] * 255, 3)
                if final_faces != []:
                    cv2.putText(img, 'ID : %d ' % (d[4]), (d[0] - 10, d[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                colours[d[4] % 32, :] * 255, 2)                                    
                else:
                    cv2.putText(img, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                colours[d[4] % 32, :] * 255, 2)
                if global_id_dict.get(str(d[4])) is None:
                    id = None
                else:
                    id = global_id_dict.get(str(d[4]))                               
                if id != None:
                    cv2.putText(img, 'UID : ' + (str(id)), (d[0] - 10, d[1] - 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                colours[d[4] % 32, :] * 255, 2)   
            # out.write(img)

                # centers = np.asarray(centers)
                # self.tracker.update(centers)
                # for j in range(len(self.tracker.tracks)):
                #     if(len(self.tracker.tracks[j].trace)>1):
                #         x = int(self.tracker.tracks[j].trace[-1][0,0])
                #         y = int(self.tracker.tracks[j].trace[-1][0,1])
                #         # tl = (x-10,y-10)
                #         # br = (x+10,y+10)
                #         # # cv2.rectangle(rgb,tl,br,(0,0,255),1)
                #         cv2.putText(img,str(self.tracker.tracks[j].trackId), (x-10,y-20),0, 1.2, self.tracker.tracks[j].color,4)
                #                     # for k in range(len(self.tracker.tracks[j].trace)):
                #                     #     x = int(self.tracker.tracks[j].trace[k][0,0])
                #                     #     y = int(self.tracker.tracks[j].trace[k][0,1])
                #                     #     cv2.circle(rgb,(x,y), 3, self.tracker.tracks[j].color,-1)
                                    # cv2.circle(rgb,(x,y), 6, self.tracker.tracks[j].color,-1)                    
                # if len(features)!=0:
                #     simScore = self.savedFeaturs @ torch.cat(features).T
                #     # print(simScore.T.detach().numpy().tolist())
                #     print(simScore.argmax())
            tme = time.time()
            proc_time = '{:2f}'.format((tme-tms)*1000)
            # print(proc_time)
            cv2.putText(img,f"{proc_time}",(10,40),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2)
            img =cv2.resize(img,(1280,720))
            out.write(img)
            # cv2.imshow('Stream',img)
            # key = cv2.waitKey(delay)
            # if key == ord('q'):
            #     break
            # if key == ord('p'):
            #     delay = 0
            # if key == ord('r'):
            #     delay = 30
        results = []
        out.release()

def help():
    parser = ArgumentParser()
    parser.add_argument('-c','--config',help='Config json file',required=True)
    parser.add_argument('-i','--input',help='Input image/folder/video/rtsp',required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = help()
    config = json.load(open(args.config))

    pipeline = VideoAnalytics(config,args.input)
    pipeline.runDetectionPipeline()

        

        
