import cv2
import numpy as np
from typing import List,Tuple,Dict
import torch 
import os
import torchvision.transforms as transforms
from .Res18Unet import ResNet18Segmentation as R18U

class IR:
    def __init__(
            self,
            image_path: str,
            model_weights_path: str,
            ROI: List[int] =None,
    )-> None:
        assert ROI and len(ROI) == 4, "Invalid ROI"
        assert os.path.exists(model_weights_path),f"Invalid model_weights_path: {model_weights_path}"
        self.ROI=ROI
        self.original_img=None
        self.img=None
        self.ROI_img=None
        self.hsv_img=None
        self.image_path = None

        self.model=R18U(num_classes=2,pretrained=False)
        state_dict = torch.load(model_weights_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.load_image(image_path)
        del state_dict

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
        h, w = self.original_img.shape[:2]
        assert 0 <= x0 < x1 < w and 0 <= y0 < y1 < h, "Invalid ROI"
        
        self.ROI_img = self.original_img[y0:y1, x0:x1]
        self.hsv_img = cv2.cvtColor(self.ROI_img, cv2.COLOR_BGR2HSV)
    
    def load_image(
            self,
            image_path: str,
    )-> None:
        assert image_path and isinstance(image_path, str), "Invalid image_path"
        self.original_img = cv2.imread(image_path)
        assert self.original_img is not None, f"Failed to load {image_path}"
        self.img=self.original_img.copy()
        self.image_path=image_path
        self._extract_roi()

    def get_results(
            self,
            min_area: int =1000,
    )-> Dict:
        mask=self._get_mask()
        contours=self._get_contours(mask,min_area)
        bboxes,angles=self._get_bboxes(contours)
        cv2.rectangle(self.img, (self.ROI[0],self.ROI[1]),(self.ROI[2],self.ROI[3]),(255,255,255),2)
        cv2.putText(self.img,"ROI",(self.ROI[0],self.ROI[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        offset_x,offset_y=self.ROI[:2]
        for x,y,w,h in bboxes:
            x+=offset_x
            y+=offset_y
            cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 255, 255), 1)
        idx,Max_Area=self._get_max_area(contours)
        if idx<0:
            result={}
            result['Num']=0
            result['W']=0
            result['H']=0
            result['Rect_Ratio']=0
            result['Area']=Max_Area
            result['Angle']=0
            result['Convexity_Ratio']=0
            result['Uniformity']=0
            return result
        
        max_cnt=contours[idx]
        max_mask = np.zeros(self.hsv_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(max_mask, [max_cnt], -1, 255, -1)
        Uniformity=self._calculate_color_uniformity(max_mask,self.hsv_img)
        
        x, y, W, H = bboxes[idx]
        Rect_Ratio=Max_Area/(W*H)
        Num=len(contours)
        Convexity_Ratio=self._convexity_ratio(contours[idx])
        Curvature_Smoothness = self._curvature_smoothness(max_cnt)

        result={}
        result['Num']=Num
        result['W']=W
        result['H']=H
        result['Rect_Ratio']=Rect_Ratio
        result['Area']=Max_Area
        result['Angle']=min(abs(90.0-abs(angles[idx])),abs(angles[idx]))
        result['Smoothness']=Convexity_Ratio*0.4+Rect_Ratio*0.3+Curvature_Smoothness*0.3
        result['Uniformity']=Uniformity
        return result
    
    @torch.no_grad
    def _get_mask(
            self
    )-> np.ndarray:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        rgb=cv2.cvtColor(self.ROI_img, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(rgb)
        img_normalized = normalize(img_tensor).unsqueeze(0)
        pred=self.model(img_normalized)
        pred_mask = torch.argmax(pred, dim=1)
        binary_img = (pred_mask == 1).squeeze(0).cpu().numpy().astype(np.uint8) * 255
        return binary_img
    
    def _calculate_color_uniformity(
            self,
            mask: np.ndarray,
            hsv_img: np.ndarray,
    )-> float:
        h = hsv_img[..., 0][mask > 0]
        h_rad = np.deg2rad(h.astype(np.float32) * 2)
        r = np.hypot(np.cos(h_rad).mean(), np.sin(h_rad).mean())
        return float(np.exp(-np.sqrt(-2 * np.log(r + 1e-8))))
    
    def _curvature_smoothness(
            self, 
            contour: np.ndarray
    )-> float:
        """计算边界曲率光滑度"""
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
    
        if len(approx) < 5:
            return 1.0
        curvatures = []
        pts = approx.squeeze()
        n = len(pts)
    
        for i in range(n):
            p_prev = pts[(i-1) % n]
            p_curr = pts[i]
            p_next = pts[(i+1) % n]
        
            # 向量夹角变化
            v1 = p_prev - p_curr
            v2 = p_next - p_curr
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            curvatures.append(angle)
    
        # 曲率变化的标准差（越小越光滑）
        curvature_std = np.std(curvatures)
        smoothness = np.exp(-curvature_std * 5)  # 映射到[0,1]
        return smoothness
        

    def _convexity_ratio(
            self,
            contour: np.ndarray
    )-> float:
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour) # 凸包
        hull_area = cv2.contourArea(hull)
        # 比值越接近1越光滑
        ratio = area / hull_area if hull_area > 0 else 0
        return min(1,ratio)

    def _get_max_area(
            self,
            contours: List,
    )-> Tuple:
        idx=-1
        max_area=-1
        for i,cnt in enumerate(contours):
            temp=cv2.contourArea(cnt)
            if temp>max_area:
                idx=i
                max_area=temp
        return idx,max_area

    def _get_bboxes(
            self,
            contours: Tuple,
    )-> Tuple[List]:
        bboxes,angles = [],[]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            angles.append(angle)
            bboxes.append((x, y, w, h))
        return bboxes,angles

    def _get_contours(
            self,
            mask: np.ndarray,
            min_area: int =1000,
            use_close: bool =True,
    ) ->List[np.ndarray]:
        if use_close:
            kernel_close = np.ones((6,6), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        f_contours=[cnt for cnt in contours if cv2.contourArea(cnt)>=min_area]
        return f_contours
    
    
    