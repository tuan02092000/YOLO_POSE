from yolov8.ultralytics.models.yolo import YOLO

class PoseYOLO():
    def __init__(self, cfg):
        self.cfg = cfg
        self.DEVICE = cfg['DEVICE']
        self.WEIGHT = cfg['MODEL_POSE']['WEIGHT']
        self.IMGSZ = cfg['MODEL_POSE']['IMGSZ']
        self.IOU = cfg['MODEL_POSE']['IOU']
        self.CONF = cfg['MODEL_POSE']['CONF']
        self.CLASS_IDX = cfg['MODEL_POSE']['CLASS_IDX']
        self.CLASS_NAMES = cfg['MODEL_POSE']['CLASS_NAMES']
        self.PERSIST = cfg['MODEL_POSE']['PERSIST']
        self.FORMAT = cfg['MODEL_POSE']['FORMAT']
        self.HALF = cfg['MODEL_POSE']['HALF']
        self.INT8 = cfg['MODEL_POSE']['INT8']
        self.TRACKER = cfg['MODEL_POSE']['TRACKER']

        self.STREAM = cfg['CAMERA']['STREAM']

        self.model = self.load_model()
    def load_model(self):
        return YOLO(self.WEIGHT)
    def export_model(self):
        self.model.export(format=self.FORMAT, half=self.HALF, int8=self.INT8)
    def inference_tracking(self, frame):
        results = self.model.track(frame, device=self.DEVICE, persist=self.PERSIST, stream=self.STREAM, tracker=self.TRACKER, iou=self.IOU, conf=self.CONF)
        return results
