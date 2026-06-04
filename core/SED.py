import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO
from typing import List,Tuple,Dict

class SED:
    def __init__(
            self,
            image_path: str,
            model_weights_path: str,
            ROI: List[int],
            target_size: Tuple[int,int] =(1280,1024),
    )->None :
        assert ROI and len(ROI) == 4, "Invalid ROI"
        assert len(target_size)==2 and target_size[0]>0 and target_size[1]>0,"Invalid target_size"
        assert os.path.exists(model_weights_path),f"Invalid model_weights_path: {model_weights_path}"
        self.ROI=ROI
        self.original_img=None
        self.img=None
        self.ROI_img=None
        self.image_path = None
        self.target_size=target_size
        self.load_image(image_path)

        self.model=YOLO(model_weights_path)
    
    def reset_ROI(
            self,
            ROI: List[int] =None,
    )-> None:
        assert ROI and len(ROI) == 4, "Invalid ROI"
        assert self.original_img and self.image_path, "Please load_image first"
        self.ROI=ROI
        self._extract_roi()
    
    def _extract_roi(self) -> None:
        x0, y0, x1, y1 = self.ROI
        h, w = self.img.shape[:2]
        assert 0 <= x0 < x1 <=w and 0 <= y0 < y1 <=h, "Invalid ROI"
        self.ROI_img = self.img[y0:y1, x0:x1]

    def load_image(
            self,
            image_path: str,
    )-> None:
        assert image_path and isinstance(image_path, str), "Invalid image_path"
        self.original_img = cv2.imread(image_path)
        assert self.original_img is not None, f"Failed to load {image_path}"
        self.image_path=image_path
        self.img=self.original_img.copy()
        self.img = cv2.resize(self.img,self.target_size,interpolation=cv2.INTER_LINEAR)
        self._extract_roi()

    def get_yolo_results(
            self,
            conf_threshold: float=0.4,
            sed_conf: float=0.4,
            fod_conf: float=0.7,
    )->Dict:
        bboxes,confs,classes = [],[],[]
        yolo_result=self.model.predict(
            self.ROI_img, iou=0.4,
            conf=conf_threshold,
            imgsz=(640,1184))[0]
        if yolo_result.boxes:
            confs=yolo_result.boxes.conf.cpu().numpy().tolist()
            classes=yolo_result.boxes.cls.cpu().numpy().tolist()
            xywh=yolo_result.boxes.xywh.cpu().numpy()
            for box in xywh:
                cx, cy, w, h = box
                x = cx - w / 2
                y = cy - h / 2
                bboxes.append([int(x), int(y), int(w), int(h)])  

        cv2.rectangle(self.img, (self.ROI[0],self.ROI[1]),(self.ROI[2],self.ROI[3]),(255,255,255),2)
        cv2.putText(self.img,"ROI",(self.ROI[0],self.ROI[1]+15), cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255),2)
        results={}
        SE_Confs,FO_Confs=[],[]
        SE_count,FO_count=0,0
        offset_x,offset_y=self.ROI[:2]

        for i,conf in enumerate(confs):
            x,y,w,h=bboxes[i]
            x+=offset_x
            y+=offset_y
            if classes[i]==0 and conf>=sed_conf:
                SE_count+=1
                SE_Confs.append(conf)
                cv2.rectangle(self.img, (x,y), (x+w,y+h), (0, 255, 0), 2)
                cv2.putText(self.img, f"SED {conf:.2f}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 3)
            elif classes[i]==1 and conf>=fod_conf:
                FO_count+=1
                FO_Confs.append(conf)
                cv2.rectangle(self.img, (x,y), (x+w,y+h), (0, 0, 255), 2)
                cv2.putText(self.img, f"FOD {conf:.2f}", (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3)
        results["SE_Num"]=SE_count
        results["FO_Num"]=FO_count
        results["SE_Confs"]=SE_Confs
        results["FO_Confs"]=FO_Confs

        return results
    
        

    


    