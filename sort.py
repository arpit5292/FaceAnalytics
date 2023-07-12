

# import lib.utils as utils
import numpy as np
import os
from data_association import associate_detections_to_trackers
from kalman_tracker import KalmanBoxTracker
from FaceFeatures import AdaFeatures



class Sort:

    def __init__(self):
        """
        Sets key parameters for SORT
        """
        self.max_age = 1
        self.min_hits = 5
        # self.dist_rate_threshold = config["dist_rate_threshold"]
        # self.high_ratio_variance = config["high_ratio_variance"]        
        self.predict_num = 5
        self.iou_threshold = 0.15
        self.trackers = []
        self.frame_count = 0
        self.global_id_dict = {}
        self.metaPipeline = []
        self._createMetaEngines()        
        self.min_faces_in_tracks = 1
    
    def _createMetaEngines(self):
        # for x in self.analyticsConfig['metadata']:
        self.metaPipeline.append(AdaFeatures('ir_50','./data/weights/'))

    def update(self,dets,img_size,addtional_attribute_list,savedFeaturs,savedId):
        """
        Params:
          dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE:as in practical realtime MOT, the detector doesn't run on every single frame
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()  # kalman predict ,very fast ,<1ms
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if dets != []:

            matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks,self.iou_threshold)

            # update matched trackers with assigned detections
            for t, trk in enumerate(self.trackers):
                if t not in unmatched_trks:
                    d = matched[np.where(matched[:, 1] == t)[0], 0]
                    trk.update(dets[d, :][0])
                    # if addtional_attribute_list[d[0]][2] < self.dist_rate_threshold  and addtional_attribute_list[d[0]][3] < self.high_ratio_variance:                        
                    trk.face_addtional_attribute.append(addtional_attribute_list[d[0]])

            # # create and initialise new trackers for unmatched detections
            for i in unmatched_dets:
                trk = KalmanBoxTracker(dets[i, :])
                # if addtional_attribute_list[i][2] < self.dist_rate_threshold  and addtional_attribute_list[i][3] < self.high_ratio_variance:                        
                trk.face_addtional_attribute.append(addtional_attribute_list[i])
                # logger.info("new Tracker: {0}".format(trk.id + 1))
                self.trackers.append(trk)

        i = len(self.trackers)
        # tracks_to_be_saved = []
        # expid = len(os.listdir('run1')) + 1
        for trk in reversed(self.trackers):
            if dets == []:
                trk.update([])
            d = trk.get_state()
            if len(trk.face_addtional_attribute) >= self.min_faces_in_tracks and self.global_id_dict.get(str(trk.id + 1)) is None:
                for met in self.metaPipeline:
                    embd =met.runInference(trk.face_addtional_attribute[-1])
                    sim = savedFeaturs @ embd.T
                    idx = sim.argmax()
                    # print(sim.T.detach().numpy().tolist())
                    # print(self.savedId)
                    prob =sim[idx].detach().numpy()[0] 
                    subId = savedId[idx]
                    # features.append(embd)
                    # print(prob,subId)
                    # if prob<=0.20:
                        # cv2.putText(img,"{:.2f}".format(prob)+f'-{subId}',(bbox[0][0]+10,bbox[0][1]+20),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),2)
                    if prob>0.20: 
                        self.global_id_dict[str(trk.id + 1)] = subId
                        break               
                # global_id_dict = utils_obj.save_to_file(trk,logger,expid,global_id_dict)

            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                if self.global_id_dict.get(str(trk.id + 1)) is None:
                    id = None
                else:
                    id = self.global_id_dict.get(str(trk.id + 1))
                # print(id)
                # if id == 'NA':
                #     id = None 
                # print(id)                           
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update >= self.max_age or trk.predict_num >= self.predict_num or d[2] < 0 or d[3] < 0 or d[0] > img_size[1] or d[1] > img_size[0]:
                self.trackers.pop(i)

                    # tracks_to_be_saved.append(trk)
                    # logger.info('remove tracker: {0}'.format(trk.id + 1))
                # logger.info('remove tracker: {0}'.format(trk.id + 1))
                # 
        if len(ret) > 0:
            return np.concatenate(ret),self.global_id_dict
        return np.empty((0, 5)),self.global_id_dict


    # def clear_all_existing_tracks(self):
    #     i = len(self.trackers)
    #     tracks_to_be_saved = []
    #     for trk in reversed(self.trackers):
    #         i -= 1
    #         # remove dead tracklet
    #         if len(trk.face_addtional_attribute) >= self.min_faces_in_tracks:
    #             tracks_to_be_saved.append(trk)
    #         self.trackers.pop(i)
    #     return tracks_to_be_saved

