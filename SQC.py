import json
import cv2
import os
import numpy as np
import core.IR as IR
import core.OCR as OCR
import core.SED as SED
from typing import List,Tuple,Dict
R18U_PATH="./checkpoint/Res18Unet_fine_tuning.pth"
YOLO_PATH="./checkpoint/yolov8n_best.pt"
CONF_PATH="./config/config.json"


class SQC:
    def __init__(
            self,
            IR_image_path: str,
            SED_image_path: str
    )-> None:
        assert os.path.exists(IR_image_path),f"Invalid image_path: {IR_image_path}"
        assert os.path.exists(SED_image_path),f"Invalid image_path: {SED_image_path}"
        assert os.path.exists(CONF_PATH),f"Invalid image_path: {CONF_PATH}"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, CONF_PATH)
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"Invalid config_path: {config_path}")
        self.config_path=config_path
        self.config=config
        self.IR_cfg=config.get("IR",{})
        self.SED_cfg=config.get("SED",{})
        self.OCR_cfg=config.get("OCR",{})
        self.check=config.get("check",{})
        assert (
            self.config!={} and 
            self.IR_cfg!={} and 
            self.SED_cfg!={} and 
            self.OCR_cfg!={} and 
            self.check!={}
            ),"Invalid config"
        self.IR=IR.IR(
            IR_image_path,
            self.IR_cfg.get("model_path",R18U_PATH),
            self.IR_cfg.get("ROI",[20,20,390,140])
        )
        self.OCR=OCR.OCR(
            IR_image_path,
            self.OCR_cfg.get("ROI",[0,275,175,299])
        )
        self.SED=SED.SED(
            SED_image_path,
            self.SED_cfg.get("model_path",YOLO_PATH),
            self.SED_cfg.get("ROI",[100,200,1000,400]),
            tuple(self.SED_cfg.get("classes_name",("SED","FOD")))
        )

        self.IR_and_OCR_result={}
        self.SED_result={}
        self.check_IR_and_OCR_result={}
        self.check_SED_result={}

    def load_image(
            self,
            IR_image_path: str,
            SED_image_path: str
    )-> None:
        assert os.path.exists(IR_image_path),f"Invalid image_path: {IR_image_path}"
        assert os.path.exists(SED_image_path),f"Invalid image_path: {SED_image_path}"
        self.IR.load_image(IR_image_path)
        self.OCR.load_image(IR_image_path)
        self.SED.load_image(SED_image_path)
        self.IR_and_OCR_result={}
        self.SED_result={}
        self.check_IR_and_OCR_result={}
        self.check_SED_result={}

    def sed(self)-> Dict:
        """binary_threshold=int(self.SED_cfg.get("binary_threshold",50))
        score_threshold=self.SED_cfg.get("score_threshold",0.6)
        close_kernel=tuple(self.SED_cfg.get("close_kernel",(7,7)))
        min_area=int(self.SED_cfg.get("min_area",20))
        #self.SED_result=self.SED.get_results(binary_threshold,score_threshold,close_kernel,min_area)"""
        self.SED_result=self.SED.get_yolo_results(
            self.SED_cfg.get("iou",0.4),
            self.SED_cfg.get("sed_conf",0.4),
            self.SED_cfg.get("fod_conf",0.7)
        )
        return self.SED_result
    
    def ir_and_ocr(self)-> Dict:
        IR_min_area=int(self.IR_cfg.get("min_area",1000))
        IR_result=self.IR.get_results(IR_min_area)

        OCR_min_area=int(self.OCR_cfg.get("min_area",0))
        wh_ratio=tuple(self.OCR_cfg.get("wh_ratio",(0.2,5.0)))
        template_dir=self.OCR_cfg.get("template_dir","./utils/template")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_dir = os.path.join(current_dir, template_dir)
        OCR_result=self.OCR.get_temperature(OCR_min_area,wh_ratio,template_dir)
        self.IR_and_OCR_result=IR_result|OCR_result
        return self.IR_and_OCR_result
    
    @property
    def show_sed_img(self)-> np.ndarray:
        """Return BGR img"""
        return self.SED.show_img
    @property
    def show_ir_and_ocr_img(self)-> np.ndarray:
        """Return BGR img"""
        return self.IR.img
    
    def check_all(self)->None:
        self.check_sed()
        self.check_ir_and_ocr()

    def check_sed(self)-> None:
        if self.SED_result=={}:
            self.sed()

        temp=self.SED_result["SE_Num"]
        if temp>0:
            self.check_SED_result["status_sed"]="NG"
            self.check_SED_result["reason_sed"]=f"we need <SE_Num=0> but find :{temp}"
            return 
        
        temp=self.SED_result["FO_Num"]
        if temp>0:
            self.check_SED_result["status_sed"]="NG"
            self.check_SED_result["reason_sed"]=f"we need <FO_Num=0> but find :{temp}"
            return 
        
        self.check_SED_result["status_sed"]="OK"

    def check_ir_and_ocr(self)->None:
        if self.IR_and_OCR_result=={}:
            self.ir_and_ocr()

        Num=self.check.get("Num")
        temp=self.IR_and_OCR_result["Num"]
        if temp!=Num:
            self.check_IR_and_OCR_result["status_ir_and_ocr"]="NG"
            self.check_IR_and_OCR_result["reason_ir_and_ocr"]=f"we need <Num={Num}> but find :{temp}"
            return 
        
        W=self.check.get("W")
        temp=self.IR_and_OCR_result["W"]
        if not (W[0]<=temp<=W[1]):
            self.check_IR_and_OCR_result["status_ir_and_ocr"]="NG"
            self.check_IR_and_OCR_result["reason_ir_and_ocr"]=f"we need <{W[0]} <= W <= {W[1]}> but find :{temp}"
            return
        
        H=self.check.get("H")
        temp=self.IR_and_OCR_result["H"]
        if not (H[0]<=temp<=H[1]):
            self.check_IR_and_OCR_result["status_ir_and_ocr"]="NG"
            self.check_IR_and_OCR_result["reason_ir_and_ocr"]=f"we need <{H[0]} <= H <= {H[1]}> but find :{temp}"
            return
        
        # Rect_Ratio = self.check.get("Rect_Ratio")
        # temp = self.IR_and_OCR_result["Rect_Ratio"]
        # if temp < Rect_Ratio:
        #     self.check_IR_and_OCR_result["status_ir_and_ocr"] = "NG"
        #     self.check_IR_and_OCR_result["reason_ir_and_ocr"] = f"we need <Rect_Ratio >= {Rect_Ratio}> but find: {temp:.2f}"
        #     return
        
        Uniformity = self.check.get("Uniformity")
        temp=self.IR_and_OCR_result["Uniformity"]
        if temp < Uniformity:
            self.check_IR_and_OCR_result["status_ir_and_ocr"] = "NG"
            self.check_IR_and_OCR_result["reason_ir_and_ocr"] = f"we need <Uniformity >= {Uniformity}> but find: {temp:.2f}"
            return
        
        Area = self.check.get("Area")
        temp = self.IR_and_OCR_result["Area"]
        if not (Area[0] <= temp <= Area[1]):
            self.check_IR_and_OCR_result["status_ir_and_ocr"] = "NG"
            self.check_IR_and_OCR_result["reason_ir_and_ocr"] = f"we need <{Area[0]} <= Area <= {Area[1]}> but find: {temp:.2f}"
            return
        
        Angle = self.check.get("Angle")
        temp = self.IR_and_OCR_result["Angle"]
        if temp > Angle:
            self.check_IR_and_OCR_result["status_ir_and_ocr"] = "NG"
            self.check_IR_and_OCR_result["reason_ir_and_ocr"] = f"we need <Angle <= {Angle}> but find: {temp:.2f}"
            return
        
        Smoothness = self.check.get("Smoothness")
        temp = self.IR_and_OCR_result["Smoothness"]
        if temp < Smoothness:
            self.check_IR_and_OCR_result["status_ir_and_ocr"] = "NG"
            self.check_IR_and_OCR_result["reason_ir_and_ocr"] = f"we need <Smoothness >= {Smoothness}> but find: {temp:.2f}"
            return
        
        # High_Temperature = self.check.get("High_Temperature")
        # temp=float((self.IR_and_OCR_result["High_Temperature"]).split()[0])
        # if not (High_Temperature[0] <= temp <= High_Temperature[1]):
        #     self.check_IR_and_OCR_result["status_ir_and_ocr"] = "NG"
        #     self.check_IR_and_OCR_result["reason_ir_and_ocr"] = f"we need <{High_Temperature[0]} <= High_Temperature <= {High_Temperature[1]}> but find: {temp}"
        #     return
        
        # Low_Temperature = self.check.get("Low_Temperature")
        # temp=float((self.IR_and_OCR_result["Low_Temperature"]).split()[0])
        # if not (Low_Temperature[0] <= temp <= Low_Temperature[1]):
        #     self.check_IR_and_OCR_result["status_ir_and_ocr"] = "NG"
        #     self.check_IR_and_OCR_result["reason_ir_and_ocr"] = f"we need <{Low_Temperature[0]} <= Low_Temperature <= {Low_Temperature[1]}> but find: {temp}"
        #     return
        
        self.check_IR_and_OCR_result["status_ir_and_ocr"] = "OK"
        

        
        
        


        
