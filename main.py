import yaml
from src.pose.pose_estimation import PoseYOLO
from src.action.action_recognition import ActionModel
from uitls.process import process_stream

if __name__ == '__main__':
    with open("cfg/base.yaml", 'r') as stream:
        cfg = yaml.safe_load(stream)
    pose_model = PoseYOLO(cfg)
    action_model = ActionModel(cfg)
    process_stream(cfg, pose_model, action_model)

