import os
import time
import threading
import queue
from pathlib import Path

"""
模拟相机传入，使用多线程
"""

class ImageLoader:
    def __init__(
            self,
            sed_dir: str,
            ir_dir: str,
            window_size: float =0.1,
    )-> None:
        assert Path(ir_dir).is_dir(), f"Invalid directory: {ir_dir}"
        assert Path(sed_dir).is_dir(), f"Invalid directory: {sed_dir}"
        self.sed_dir=sed_dir
        self.ir_dir = ir_dir
        self.window_size = window_size

        self.sed_queue=queue.Queue()
        self.ir_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.running = False
        self.sed_thread = None
        self.ir_thread = None
        self.matcher_thread = None
    
    def _producer(
            self,
            img_dir: str,
            out_queue: queue.Queue,
    )-> None:
        valid_exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        try:
            images = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
        except Exception:
            return
        
        for img_name in images:
            if not self.running:
                break
            img_path=os.path.join(img_dir,img_name)
            time.sleep(0.1) # 0.1sec read once
            timestamp=time.time()
            out_queue.put((timestamp, img_name, img_path))
    
    def _matcher(
            self
    )->None:
        sed_buffer = []
        ir_buffer = []
        while self.running:
            while True:
                try:
                    sed_buffer.append(self.sed_queue.get_nowait())
                except queue.Empty:
                    break
            while True:
                try:
                    ir_buffer.append(self.ir_queue.get_nowait())
                except queue.Empty:
                    break
            
            matched_sed_indices=[]
            matched_ir_indices=[]
            for i,sed_item in enumerate(sed_buffer):
                if i in matched_sed_indices:
                    continue
                sed_time=sed_item[0]
                closest_j=-1
                min_diff=float("inf")
                for j,ir_item in enumerate(ir_buffer):
                    if j in matched_ir_indices:
                        continue
                    ir_time=ir_item[0]
                    diff=abs(sed_time-ir_time)
                    if diff < min_diff:
                        min_diff=diff
                        closest_j=j
                if closest_j != -1 and min_diff<= self.window_size:
                    matched_sed_indices.append(i)
                    matched_ir_indices.append(closest_j)
                    self.result_queue.put((sed_item, ir_buffer[closest_j]))
            
            # 未匹配的
            sed_buffer=[item for i,item in enumerate(sed_buffer) if i not in matched_sed_indices]
            ir_buffer =[item for i, item in enumerate(ir_buffer) if i not in matched_ir_indices]
            # 清理过期数据(5.0sec)
            current_time = time.time()
            sed_buffer = [item for item in sed_buffer if current_time - item[0] <= 5.0]
            ir_buffer = [item for item in ir_buffer if current_time - item[0] <= 5.0]

            time.sleep(0.05)
    
    def start(self)-> None:
        self.running=True
        self.sed_thread=threading.Thread(
            target=self._producer,
            args=(self.sed_dir, self.sed_queue),
            daemon=True
        )
        self.ir_thread=threading.Thread(
            target=self._producer,
            args=(self.ir_dir, self.ir_queue),
            daemon=True
        )
        self.matcher_thread=threading.Thread(
            target=self._matcher,
            daemon=True
        )
        self.sed_thread.start()
        self.ir_thread.start()
        self.matcher_thread.start()

    def stop(self):
        self.running=False
        if self.sed_thread and self.sed_thread.is_alive():
            self.sed_thread.join(timeout=1.0)
        if self.ir_thread and self.ir_thread.is_alive():
            self.ir_thread.join(timeout=1.0)
        if self.matcher_thread and self.matcher_thread.is_alive():
            self.matcher_thread.join(timeout=1.0)



