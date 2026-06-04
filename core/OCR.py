import cv2
import numpy as np
import os
import re
from utils.Template import TemplateMatcher
from typing import List,Tuple,Dict

TEMPLATE_DIR = r"./template"  

class OCR:
    def __init__(
            self,
            image_path: str,
            ROI: List[int] =None,
    )-> None:
        assert ROI and len(ROI) == 4, "Invalid ROI"
        self.ROI=ROI
        self.original_img=None
        self.img=None
        self.ROI_img=None
        self.hsv_img=None
        self.image_path = None
        self.load_image(image_path)

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
        self.image_path=image_path
        self.img=self.original_img.copy()
        self._extract_roi()
    
    def auto_template(
            self,
            img: np.ndarray,
            text_blocks: List[np.ndarray],
            save_path: str
    )-> None:
        img=img.copy()
        if len(img.shape)==3:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for i,(x,y,w,h) in enumerate(text_blocks):
            temp=img[y:y+h,x:x+w]
            cv2.imwrite(os.path.join(save_path, f"{i}.png"), temp)
        

    def get_temperature(
            self,
            min_area: int =0,
            wh_ratio: Tuple[float,float] =(0.2,5.0),
            template_dir: str =TEMPLATE_DIR,
    )-> Dict:
        h_texts=self.get_text(min_area,wh_ratio,'red',template_dir)
        l_texts=self.get_text(min_area,wh_ratio,'green',template_dir)
        High_Temperature=self._text_2_temperature(h_texts)
        Low_Temperature=self._text_2_temperature(l_texts)
        result={}
        result["High_Temperature"]=High_Temperature
        result["Low_Temperature"]=Low_Temperature
        return result
    
    def _text_2_temperature(
            self,
            texts: List,
    )-> str:
        if len(texts)==0:
            return "NO INFO"
        temperature,orignal_texts="",""
        for char in texts:
            orignal_texts+=char
        numbers = re.findall(r'\d+',orignal_texts)
        if len(numbers)==0:
            return "-1 can't analysis "
        for i,num in enumerate(numbers):
            temperature+=num
            temperature+="."
        return temperature[:-1]+" ℃ "
        

    def get_text(
            self,
            min_area: int =50,
            wh_ratio: Tuple[float,float] =(0.2,5.0),
            color: str ='red',
            template_dir: str =TEMPLATE_DIR,
    )-> List:
        min_area=max(0,min_area)
        assert color in ('red','green')," Color must in (red,green) "
        mask=self._get_color_mask(self.hsv_img,color)
        text_blocks=self._get_text_blocks(mask,min_area,wh_ratio)
        matcher=self._get_matcher(template_dir)
        assert matcher," Matcher error "
        texts,_=self._match(text_blocks,mask,matcher)
        return texts

    def _match(
            self,
            text_blocks: List[np.ndarray],
            mask: np.ndarray,
            matcher: TemplateMatcher
    )-> Tuple[List,List]:
        labels,scores=[],[]
        for x,y,w,h in text_blocks:
            block_mask = mask[y:y+h,x:x+w]
            label, score = matcher.predict(block_mask)
            if label == 'unknown':
                label='-'
            elif label == 'point':
                label='.'
            labels.append(label)
            scores.append(score)
        return labels,scores

    def _get_matcher(
            self,
            template_dir: str =TEMPLATE_DIR,
    ):
        try:
            matcher = TemplateMatcher(
                template_dir=template_dir,
                img_size=(40, 80),
                method="template",
                threshold=0.5
            )
            return matcher
        except ValueError as e:
            print(f"初始化匹配器失败：{e}\n")
            return None

    def _get_text_blocks(
            self,
            mask: np.ndarray,
            min_area: int =50,
            wh_ratio: Tuple[float,float] =(0.2,5.0)
    )-> List[np.ndarray]:
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        text_blocks=[]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            ratio = w / h
            if wh_ratio[0]<ratio < wh_ratio[1] and w * h >= min_area:
                text_blocks.append((x, y, w, h))
        text_blocks = sorted(text_blocks, key=lambda b: b[0])
        return text_blocks

    def _get_color_mask(
            self,
            hsv_img: np.ndarray,
            color_name: str='green',
    )-> np.ndarray:
        """
        Args:
            color_name: white, red, orange, green
        Returns:
            mask
        """
        if color_name == 'white':
            mask = cv2.inRange(hsv_img, np.array([0, 0, 200]), np.array([180, 30, 255]))
        elif color_name == 'red':
            mask1 = cv2.inRange(hsv_img, np.array([0, 50, 50]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv_img, np.array([156, 50, 50]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        elif color_name == 'orange':
            mask = cv2.inRange(hsv_img, np.array([10, 50, 50]), np.array([25, 255, 255]))
        elif color_name == 'green':
            mask = cv2.inRange(hsv_img, np.array([35, 50, 50]), np.array([85, 255, 255]))
        else:
            raise ValueError("color must in (white, red, orange, green)")
        return mask
    

