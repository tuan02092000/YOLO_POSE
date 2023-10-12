import cv2
import numpy as np
from yolov8.ultralytics.utils.plotting import Annotator

class ObjTracking():
    def __init__(self, id, keypoint, box):
        self.id = id
        self.miss_frame = 0
        self.count_falldown = 0
        self.keypoint_current = keypoint
        self.box_current = box
        self.keypoint_list = []
        self.keypoint_list.append(keypoint)
        self.action = 'pending...'
        self.color = [0, 255, 0]
    def reset_all(self):
        self.miss_frame = 0
        self.count_falldown = 0
        self.keypoint_list = []
    def reset_miss_frame(self):
        self.miss_frame = 0
    def reset_count_falldown(self):
        self.count_falldown = 0
    def reset_keypoint_list(self):
        self.keypoint_list = []
    def increase_miss_frame(self):
        self.miss_frame += 1
    def increase_count_falldown(self):
        self.count_falldown += 1
    def append_keypoint_list(self, kpts):
        self.keypoint_list.append(kpts)
    def change_kpts_box_current(self, keypoint, box):
        self.keypoint_current = keypoint
        self.box_current = box
    def change_action(self, action):
        self.action = action

class ListObjTracking():
    def __init__(self, cfg):
        self.list_obj_tracking = dict()
        self.IMGSZ = cfg['MODEL_POSE']['IMGSZ']

        self.MAX_MISS_FRAME = cfg['MODEL_ACTION']['MAX_MISS_FRAME']
        self.MAX_LEN_KEYPOINT = cfg['MODEL_ACTION']['MAX_LEN_KEYPOINT']
        self.THRESH = cfg['MODEL_ACTION']['THRESH']
        self.COLOR_FALLDOWN = cfg['MODEL_ACTION']['COLOR_FALLDOWN']
        self.DELETE_OBJ = cfg['MODEL_ACTION']['DELETE']

        self.COLOR_DEFAUT = cfg['PLOT']['COLOR_DEFAUT']
        self.THICNKESS = cfg['PLOT']['THICNKESS']
        self.FONTSCALE = cfg['PLOT']['FONTSCALE']
        self.RADIUS = cfg['PLOT']['RADIUS']
        self.LINE_WIDTH = cfg['PLOT']['LINE_WIDTH']
        
    def append(self, id, obj_tracking):
        self.list_obj_tracking[id] = obj_tracking
    def get(self, id):
        return self.list_obj_tracking[id]
    def get_keys(self):
        return self.list_obj_tracking.keys()
    def increase_miss_frame(self):
        for obj in self.list_obj_tracking.values():
            obj.increase_miss_frame()
    def check_miss_frame(self):
        for id in list(self.list_obj_tracking.keys()):
            if self.list_obj_tracking[id].miss_frame >= self.MAX_MISS_FRAME:
                del self.list_obj_tracking[id]
    def check_action(self, frame, action_model):
        buff_name = ''
        for obj in self.list_obj_tracking.values():
            if len(obj.keypoint_list) == self.MAX_LEN_KEYPOINT:
                pts = np.array(obj.keypoint_list.copy())
                out = action_model.inference(pts, frame.shape[:2])
                action_name = action_model.CLASS_NAMES[out[0].argmax()]
                obj.keypoint_list = obj.keypoint_list[1 : ]
                obj.change_action(action_name)
            if obj.action == "Fall Down":
                obj.increase_count_falldown()
                if obj.count_falldown < self.THRESH:
                    obj.action = buff_name
                else:
                    obj.color = self.COLOR_FALLDOWN
            else:
                obj.count_falldown = 0
                buff_name = obj.action
    def plot(self, frame_process):
        for id, obj in self.list_obj_tracking.items():
            text = '{} {}'.format(id, obj.action)
            frame_process = cv2.putText(frame_process, text, (obj.box_current[0], obj.box_current[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,  fontScale=self.FONTSCALE, color=self.COLOR_DEFAUT, thickness=self.THICNKESS) 
            frame_process = cv2.rectangle(frame_process, (obj.box_current[0], obj.box_current[1]), (obj.box_current[2], obj.box_current[3]), color=self.COLOR_DEFAUT, thickness=self.THICNKESS)
            anno = Annotator(frame_process, line_width=self.LINE_WIDTH)
            anno.kpts(obj.keypoint_current, shape=(self.IMGSZ, self.IMGSZ), radius=self.RADIUS, kpt_line=True)
            frame_process = anno.result()
        return frame_process