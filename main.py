from ultralytics import YOLO
from datetime import datetime
import threading
import json
import uuid
import numpy as np
import cv2
import gc
import time
import supervision as sv
from supervision.detection.core import Detections
from supervision.geometry.core import Point, Rect, Vector
import logging

my_dict = {}

logging.basicConfig(filename='info.log', level=logging.INFO, format='%(message)s')

# async def ClearMemory():
#     while True:
#         gc.collect()
#         time.sleep(30)

START1 = sv.Point(520,350)
END1 = sv.Point(1280,350)
START2 = sv.Point(1280,480)
END2 = sv.Point(520,480)

START3 = sv.Point(530,430)
END3 = sv.Point(0,430)
START4 = sv.Point(0,300)
END4 = sv.Point(530,300)

class collectInfo:
    def __init__(self, tracker_id):
        self.uuid = uuid.uuid1()
        self.tracker_id = tracker_id
        self.final_class_id = None
        self.class_id = {}
        self.time_in = 0
        self.time_out = 0

    def set_uuid(self):
        self.uuid = uuid.uuid1()

    def set_time_out(self, time_out):
        dt_object = datetime.fromtimestamp(time_out)    
        self.time_out = dt_object.strftime("%Y-%m-%d %H:%M:%S")

    def set_time_in(self, time_in):
        dt_object1 = datetime.fromtimestamp(time_in)
        self.time_in = dt_object1.strftime("%Y-%m-%d %H:%M:%S")
    
    def set_tracker_id(self, tracker_id):
        self.tracker_id = tracker_id

    def set_class_id(self):
        self.final_class_id = max(self.class_id, key=self.class_id.get)

    def confirm_class_id(self, class_id):
        if str(class_id) not in self.class_id:
            self.class_id[str(class_id)] = 1
        else:
            self.class_id[str(class_id)] += 1

    def post_info(self):
        if self.uuid == 0 or self.tracker_id == 0 or self.time_in == 0:
            return
        else:
            info = {
                "uuid": str(self.uuid),
                "tracker_id": int(self.tracker_id),
                "class_id": int(self.final_class_id),
                "time_in": str(self.time_in),
                "time_out": str(self.time_out)
            }
            
            json_str = json.dumps(info)
            logging.info(json_str)
            # print("tracker_id: ", self.tracker_id, "\t class_id: ", self.class_id, "\t time_in: ", self.time_in, "\t time_out: ", self.time_out)
            





def trigger1(self, detections: Detections, result):
        for xyxy, confidence, class_id, tracker_id in detections:
            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # we check if all four anchors of bbox are on the same side of vector
            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            # detection is partially in and partially out
            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]
            # handle new detection
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                continue

            # handle detection on the same side of the line
            if self.tracker_state.get(tracker_id) == tracker_state:
                continue

            self.tracker_state[tracker_id] = tracker_state
            if tracker_state:
                self.in_count += 1
                
            else:
                self.out_count += 1
                my_dict[tracker_id].set_time_in(time.time())

def trigger2(self, detections: Detections, result):
        for xyxy, confidence, class_id, tracker_id in detections:
            # handle detections with no tracker_id
            if tracker_id is None:
                continue

            # we check if all four anchors of bbox are on the same side of vector
            x1, y1, x2, y2 = xyxy
            anchors = [
                Point(x=x1, y=y1),
                Point(x=x1, y=y2),
                Point(x=x2, y=y1),
                Point(x=x2, y=y2),
            ]
            triggers = [self.vector.is_in(point=anchor) for anchor in anchors]

            # detection is partially in and partially out
            if len(set(triggers)) == 2:
                continue

            tracker_state = triggers[0]
            # handle new detection
            if tracker_id not in self.tracker_state:
                self.tracker_state[tracker_id] = tracker_state
                my_dict[tracker_id] = collectInfo(tracker_id)
                continue

            # handle detection on the same side of the line
            if self.tracker_state.get(tracker_id) == tracker_state:
                if tracker_id in my_dict:
                    my_dict[tracker_id].confirm_class_id(class_id)
                continue

            self.tracker_state[tracker_id] = tracker_state
            if tracker_state:
                self.in_count += 1
                my_dict[tracker_id].set_time_out(time.time())
                my_dict[tracker_id].set_class_id()
                my_dict[tracker_id].post_info()
                # del my_dict[tracker_id]
                
            else:
                self.out_count += 1

def main():
    # ClearMemory()
    model = YOLO('yolov8s.pt') # yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    line_zone1 = sv.LineZone(START1, END1)
    line_zone1_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
    )

    line_zone2 = sv.LineZone(START2, END2)
    line_zone2_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
    )

    line_zone3 = sv.LineZone(START3, END3)
    line_zone3_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
    )

    line_zone4 = sv.LineZone(START4, END4)
    line_zone4_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
    )


    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
    )

    for result in model.track(source="traffic.mp4", imgsz=320, show=True, stream=True): # traffic.mp4, test.mp4 or 0 for webcam
        frame = result.orig_img
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        labels = [
            f"#{tracker_id} {model.model.names[int(class_id)]} {confidence:.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        trigger1(line_zone1,detections=detections, result=result)
        line_zone1_annotator.annotate(frame=frame, line_counter=line_zone1)

        trigger2(line_zone2,detections=detections, result=result)
        line_zone2_annotator.annotate(frame=frame, line_counter=line_zone2)

        trigger2(line_zone4,detections=detections, result=result)
        line_zone4_annotator.annotate(frame=frame, line_counter=line_zone4)

        trigger1(line_zone3,detections=detections, result=result)
        line_zone3_annotator.annotate(frame=frame, line_counter=line_zone3)

        cv2.imshow('yolov8', frame)

        if (cv2.waitKey(30)==27):
            break


if __name__ == '__main__':
    main()
