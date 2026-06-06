import cv2
import numpy as np
import os
from typing import List,Tuple,Dict
import onnxruntime as ort

class SED:
    def __init__(
            self,
            image_path: str,
            model_weights_path: str,
            ROI: List[int],
            class_names: List[str] = ['SED', 'FOD'],
            colors: List[Tuple[int, int, int]] = [(255, 0, 0), (0, 255, 0)],
    )->None :
        assert ROI and len(ROI) == 4, "Invalid ROI"
        assert os.path.exists(model_weights_path),f"Invalid model_weights_path: {model_weights_path}"
        
        self.session = ort.InferenceSession(model_weights_path)
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        input_shape = input_info.shape # [B,C,H,W]

        self.target_size=(input_shape[3],input_shape[2])
        self.ROI=ROI
        self.class_names = class_names
        self.colors = colors
        if len(self.class_names)!=len(self.colors):
            raise ValueError(f"class_names与colors不匹配")
        
        self.load_image(image_path)
   
    """def reset_ROI(
            self,
            ROI: List[int] =None,
    )-> None:
        assert ROI and len(ROI) == 4, "Invalid ROI"
        assert self.original_img and self.image_path, "Please load_image first"
        self.ROI=ROI
        self._extract_roi()"""
    
    def _extract_roi(self, image: np.ndarray) -> np.ndarray:
        """
        裁剪ROI区域
        """
        x0, y0, x1, y1 = self.ROI
        h, w = image.shape[:2]
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"Invalid ROI: {self.ROI} for image shape {image.shape}")
        return image[y0:y1, x0:x1]

    def load_image(
            self,
            image_path: str,
    )-> None:
        """
        加载图片+ROI裁剪+resize
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Invalid image_path: {image_path}")
        self.raw_image = cv2.imread(image_path)
        self.img = self._extract_roi(self.raw_image)
        self.img = cv2.resize(self.img, self.target_size)
        self.show_img = self.img.copy()
        
    def inference(
            self, 
            iou: float = 0.25, 
            conf: List[float] = [0.3,0.7]
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行推理并绘制结果
        """
        temp = self._preprocess(self.img)
        outputs = self.session.run(None, {self.input_name: temp})
        boxes, scores, class_ids = self._nms(outputs[0], iou=iou, conf=conf)[0]
        for box, score, cls_id in zip(boxes.astype(int), scores, class_ids):
            x1, y1, x2, y2 = box
            color = self.colors[cls_id % len(self.colors)]
            cv2.rectangle(self.show_img, (x1, y1), (x2, y2), color, 2)
            label = f'{self.class_names[cls_id]}: {score:.2f}'
            cv2.putText(self.show_img, label, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return boxes, scores, class_ids

    def _preprocess(self, img: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        图像预处理，转为NCHW并归一化
        """
        temp = img.transpose(2, 0, 1)[None, ...].astype(np.float32)
        if normalize:
            temp /= 255.0
        return temp
    
    def _nms(
        self,
        outputs: np.ndarray,
        iou: float = 0.3,
        conf: List[float] = [0.4,0.7]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        非极大值抑制，支持批量
        """
        conf = np.array(conf)
        batch_size = outputs.shape[0]
        results = []
        for b in range(batch_size):
            pred = outputs[b]
            boxes = pred[:4].T  # (cx, cy, w, h)
            # 坐标转换
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
            scores = pred[4:].T
            class_ids = np.argmax(scores, axis=1)
            conf_scores = scores[np.arange(len(scores)), class_ids]
            if len(conf.shape) == 0:
                conf = np.full_like(conf_scores, conf)
            assert conf.shape[0] > class_ids.max(), f"conf长度不足以索引所有类别: conf={conf}, class_ids={class_ids}"
            mask = conf_scores > conf[class_ids]
            boxes_xyxy = boxes_xyxy[mask]
            conf_scores = conf_scores[mask]
            class_ids = class_ids[mask]
            if len(boxes_xyxy) == 0:
                results.append((np.empty((0, 4)), np.empty(0), np.empty(0)))
                continue
            keep = []
            idxs = np.argsort(-conf_scores)
            while len(idxs) > 0:
                keep.append(idxs[0])
                if len(idxs) == 1:
                    break
                ious = self._compute_iou(boxes_xyxy[idxs[0]], boxes_xyxy[idxs[1:]])
                idxs = idxs[1:][ious <= iou]
            results.append((boxes_xyxy[keep], conf_scores[keep], class_ids[keep]))
        return results

    def _compute_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        计算IoU
        """
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter
        return inter / (union + 1e-6)


    def get_yolo_results(
            self,
            iou: float=0.4,
            sed_conf: float=0.4,
            fod_conf: float=0.7,
    )->Dict:
        _,scores,class_ids=self.inference(iou=iou,conf=[sed_conf,fod_conf])
        scores=[float(s) if hasattr(s,'dtype') else s for s in scores]

        SE_Confs = [score for score, cls_id in zip(scores, class_ids) if cls_id == 0]
        FO_Confs = [score for score, cls_id in zip(scores, class_ids) if cls_id == 1]

        return {
            "SE_Num": len(SE_Confs),
            "FO_Num": len(FO_Confs),
            "SE_Confs": SE_Confs,
            "FO_Confs": FO_Confs
        }
    
        

    


    