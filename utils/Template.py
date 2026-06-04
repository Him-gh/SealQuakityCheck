import os
import cv2
import numpy as np
from typing import List,Tuple,Dict

class TemplateMatcher:
    """
    黑底白字
    """
    def __init__(
            self,
            template_dir: str,
            img_size: Tuple =(40, 60),
            method: str ="template",
            threshold: float =0.5,
    )-> None:

        self.template_dir = template_dir
        self.img_size = img_size
        self.method = method
        self.threshold = threshold
        self.templates = {}

        self._load_templates()

    def _load_templates(self):
        self.templates.clear()
        if not os.path.exists(self.template_dir):
            raise ValueError(f"模板目录不存在: {self.template_dir}")
        for label in os.listdir(self.template_dir):
            label_path = os.path.join(self.template_dir, label)
            if not os.path.isdir(label_path):
                continue
            self.templates[label] = []
            for file in os.listdir(label_path):
                if file.lower().endswith(('.jpg', '.png', '.bmp')):
                    img_path = os.path.join(label_path, file)
                    img = cv2.imread(img_path, 0)
                    if img is None:
                        continue
                    img = self._standardize(img)
                    self.templates[label].append(img)

    def _standardize(self, img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img = self._crop_foreground(img)
        img = cv2.resize(img, self.img_size,interpolation=cv2.INTER_NEAREST)
        return img

    def _crop_foreground(self, img):
        coords = np.column_stack(np.where(img > 0))
        if len(coords) == 0:
            return img
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        return img[y1:y2 + 1, x1:x2 + 1]

    def _match_template(self, img, template):
        result = cv2.matchTemplate(
            img, template, cv2.TM_CCOEFF_NORMED
        )
        return float(result.max())

    #def _match_ssim(self, img, template):
    #     return float(ssim(img, template, data_range=255))

    def predict(self, img, return_topk=1):
        img = self._standardize(img)
        results = []
        for label, template_list in self.templates.items():
            label_best_score = -1
            for template in template_list:
                if self.method == "template":
                    score = self._match_template(img, template)
                # elif self.method == "ssim":
                #     score = self._match_ssim(img, template)
                else:
                    raise ValueError("不支持的匹配方法")
                label_best_score = max(label_best_score, score)
            results.append((label, label_best_score))
        results.sort(key=lambda x: x[1], reverse=True)
        best_label, best_score = results[0]
        if best_score < self.threshold:
            return ("unknown", best_score) if return_topk == 1 else results[:return_topk]
        return (best_label, best_score) if return_topk == 1 else results[:return_topk]

    def batch_predict(self, img_list):
        return [self.predict(img) for img in img_list]

    def set_method(self, method):
        if method not in ["template", "ssim"]:
            raise ValueError("method 必须是 'template' 或 'ssim'")
        self.method = method

    def reload_templates(self):
        self._load_templates()

    def get_labels(self):
        return list(self.templates.keys())

