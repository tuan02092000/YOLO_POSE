from yolov8.action.tools.ActionsEstLoader import TSSTG

class ActionModel():
    def __init__(self, cfg):
        self.cfg = cfg
        self.DEVICE = cfg['DEVICE']
        self.WEIGHT = cfg['MODEL_ACTION']['WEIGHT']
        self.CLASS_NAMES = cfg['MODEL_ACTION']['CLASS_NAMES']

        self.model = self.load_model()
    def load_model(self):
        return TSSTG(self.WEIGHT, self.CLASS_NAMES, self.DEVICE)
    def inference(self, pts, image_size):
        return self.model.predict(pts, image_size)
