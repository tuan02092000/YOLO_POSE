import cv2
import numpy as np
from src.tracking.track import ObjTracking, ListObjTracking

def process_stream(cfg, pose_model, action_model):
    # Stream config
    stream_path = cfg['CAMERA']['STREAM']
    cap = cv2.VideoCapture(stream_path)
    FRAME_WIDTH = int(cap.get(3)) 
    FRAME_HEIHGT = int(cap.get(4)) 
    SIZE = (FRAME_WIDTH, FRAME_HEIHGT)  
    FPS = cap.get(cv2.CAP_PROP_FPS)

    # Write video
    if cfg['CAMERA']['SAVE_VIDEO']:
        writer_video_path = cfg['CAMERA']['WRITER_VIDEO_PATH']
        writer_video = cv2.VideoWriter(writer_video_path, cv2.VideoWriter_fourcc(*'MJPG'), FPS, SIZE) 
    
    # Setup 
    list_obj_tracking = ListObjTracking(cfg)
    frame_count = 0

    # Loop stream
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        frame_process = frame.copy()

        frame_count += 1

        if frame_count % cfg['CAMERA']['SKIP_FRAME'] == 0:
            frame_count = 0
        
            # Tracking results
            results = pose_model.inference_tracking(frame)
            for result in results:
                boxes_obj = result.boxes
                keypoints_obj = result.keypoints

                # print(keypoints_obj)

                ids = boxes_obj.id
                boxes = boxes_obj.xyxy
                keypoints = keypoints_obj.data

                # loop id
                if ids is not None:
                    list_id_checked = []
                    for idx, id in enumerate(ids):
                        obj_id = int(id.cpu().numpy())
                        obj_box = boxes[idx].cpu().numpy()
                        obj_keypoint = keypoints[idx].cpu().numpy()

                        x1, y1, x2, y2 = int(obj_box[0]), int(obj_box[1]), int(obj_box[2]), int(obj_box[3])

                        if obj_id not in list_obj_tracking.get_keys():
                            obj_tracking = ObjTracking(obj_id, keypoint=obj_keypoint, box=[x1, y1, x2, y2])
                            list_obj_tracking.append(obj_id, obj_tracking)
                        else:
                            list_obj_tracking.get(obj_id).change_kpts_box_current(keypoint=obj_keypoint, box=[x1, y1, x2, y2])
                            list_obj_tracking.get(obj_id).reset_miss_frame()
                            list_obj_tracking.get(obj_id).append_keypoint_list(obj_keypoint)
                        list_id_checked.append(obj_id)
                    for obj in list_obj_tracking.list_obj_tracking.values():
                        if obj.id in list_id_checked:
                            continue
                        list_obj_tracking.get(obj.id).increase_miss_frame()              
                else:
                    list_obj_tracking.increase_miss_frame()

                list_obj_tracking.check_miss_frame()
                list_obj_tracking.check_action(frame_process, action_model)
                frame_process = list_obj_tracking.plot(frame_process)

        if cfg['CAMERA']['SAVE_VIDEO']:
            writer_video.write(frame_process)

        if cfg['CAMERA']['SHOW_VIDEO']:
            cv2.imshow("Pose estimation humman YOLOv8", frame_process)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if cfg['CAMERA']['SHOW_VIDEO']:
        cv2.destroyAllWindows()

