#!/usr/bin/env python3
"""
不保存timestamp
"""
import sys
import glob
import pyautogui
import cv2
import re
import hashlib
import numpy as np
import time
import json
import yaml
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
import pyperclip


import pytesseract
import easyocr
from paddleocr import PaddleOCR

# 安全设置
pyautogui.FAILSAFE = True


class State(Enum):
    """状态枚举"""
    INIT = "初始化"
    SELECT_REGION = "选择区域"
    SCAN = "扫描"
    SCROLL = "滚动"
    MATCH_TEXT = "文字匹配"
    WAIT_DETAIL = "等待详情页"
    SAVE_IMAGES = "保存图片"
    GO_BACK = "返回主界面"
    DONE = "完成"



@dataclass
class MomentRecord:
    """朋友圈记录"""
    moment_index: int
    timestamp: str
    text: str
    images: List[str]
    #expected_images: int

class WeChatSmartCrawler:
    """微信朋友圈智能爬虫"""

    def __init__(self, config_path: str = "config.yaml", background_color=46):
        """初始化"""
        self.config = self.load_config(config_path)
        # 目录
        self.save_dir = Path(self.config["save_dir"])
        self.save_dir.mkdir(exist_ok=True)
        self.images_dir = self.save_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.debug_dir = self.save_dir / "debug"
        self.debug_dir.mkdir(exist_ok=True)

        # 状态
        self.background_color = background_color # 白天：255 晚上：46
        self.state = State.INIT
        self.region: Optional[Tuple[int, int, int, int]] = None
        self.center_x = 0
        self.current_moment_index = 0
        self.records: List[MomentRecord] = []
        self.back_button_position: Optional[Tuple[int, int]] = None
        self.text_position: Optional[Tuple[int, int]] = None

        # area
        self.posts = []
        self.scroll_count = 0
        self.curview = None
        self.record_keys = []

        self.ocr = easyocr.Reader(['ch_sim'])
        self.paddleocr = PaddleOCR(use_doc_orientation_classify=False,
                             use_doc_unwarping=False,
                             use_textline_orientation=False)
                             
        # 检查点
        self.checkpoint_file = self.save_dir / "checkpoint.json"
        if self.config["resume_from_checkpoint"]:
            self.load_checkpoint()

        print(f"✓ 智能爬虫 初始化完成")
        print(f"  保存目录: {self.save_dir}")
        print(f"  已处理: {len(self.records)} 条")

    def load_config(self, config_path: str) -> dict:
        """加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config
        except:
            return self.get_default_config()

    def get_default_config(self) -> dict:
        """默认配置"""
        return {
            "save_dir": "wechat_moments",
            "resume_from_checkpoint": True,
            "delays": {
                "click": 0.5,
                "page_load": 1.0,
                "scroll": 0.5,
                "back": 1.0,
                "between_images": 0.3,
                "ocr": 1.0,
                "hotkey": 0.3,
            },
            "scroll": {
                "amount": -5,
                "max_scrolls": 100
            },
            "image": {
                "format": "jpg",
                "quality": 95
            },
            "debug": {
                "save_screenshots": True,
                "verbose": True
            }
        }

    def select_region(self) -> Tuple[int, int, int, int]:
        """选择朋友圈区域"""
        print("\n" + "="*60)
        print("步骤 1: 选择朋友圈区域")
        print("="*60)
        print("请标记朋友圈内容区域（包含文字和缩略图）")
        print("1. 移动鼠标到区域左上角，按Enter")
        print("2. 移动鼠标到区域右下角，按Enter")
        print("="*60 + "\n")

        input("准备好后按 Enter...")

        print("\n移动鼠标到左上角...")
        input("按 Enter 确认: ")
        time.sleep(self.config["delays"]["click"])
        x1, y1 = pyautogui.position()
        print(f"✓ 左上角: ({x1}, {y1})")

        print("\n移动鼠标到右下角...")
        input("按 Enter 确认: ")
        time.sleep(self.config["delays"]["click"])
        x2, y2 = pyautogui.position()
        print(f"✓ 右下角: ({x2}, {y2})")

        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        region = (left, top, width, height)
        print(f"\n✓ 区域: 左={left}, 上={top}, 宽={width}, 高={height}")

        return region


    def find_post_area(self, region):
        # 截图
        screenshot = pyautogui.screenshot(region=region)
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        is_blank = [(w==self.background_color).all() for w in screenshot_cv]
        have_post = []

        # n = len(is_blank)
        # pairs = []
        # st = None
        # prev_val = is_blank[0]
        # run_start = 0

        # for i in range(1, n):
        #     if is_blank[i] != prev_val:
        #         run_len = i - run_start
        #         # 跳变 True->False
        #         if prev_val and not is_blank[i]:
        #             if run_len > 3:
        #                 st = i  # st 是 False 段的起点
        #         # 跳变 False->True
        #         elif not prev_val and is_blank[i]:
        #             if run_len > 3 and st is not None:
        #                 pairs.append([st, i])  # ed 是 True 段的起点
        #                 st = None
        #         run_start = i
        #         prev_val = is_blank[i]

        for i in range(len(is_blank)-1):
            if is_blank[i]==True and is_blank[i+1]==False:
                #  add post st
                have_post.append([i])
            
            if is_blank[i] == False and is_blank[i+1] == True:
                # add post ed
                if len(have_post):
                    have_post[-1].append(i)
                else:
                    # drop the first unfinished one
                    continue
                    # have_post.append([0, i])

        if len(have_post[-1]) == 1:
            # drop last unfinished one
            have_post.pop()
            # have_post[-1].append(len(is_blank))

        st = have_post[-1][0]
        ed = have_post[-1][1]
        return have_post, screenshot_cv[st:ed]


    def select_and_get_text(self):
        """选中文字并获取其内容"""
        # 自动点击文字开始
        x, y = self.text_position
        print(f"  点击首字符: ({x}, {y})")
        pyautogui.mouseDown(x, y)

        pyperclip.copy('')
        # 全选当前文本框或选中区域的文字
        pyautogui.hotkey('command', 'a')  # macOS
        
        
        time.sleep(self.config["delays"]["hotkey"])
        pyautogui.mouseUp(x, y)

        # 复制到剪贴板
        pyautogui.hotkey('command', 'c')  # macOS
        
        time.sleep(self.config["delays"]["hotkey"])
        
        # 从剪贴板获取文字
        text = pyperclip.paste()
        return text

    def get_post_time(self, region_cv, original_image):
        """
        四阶段，试了四种工具，没识别出来就存debug文件夹里
        """
        # 4. 在该区域进行OCR识别
        bottom_image = region_cv[:,30:200,:]
        text = pytesseract.image_to_string(bottom_image, lang='chi_sim+eng')
        timestamp = self.chinese_date_to_timestamp(text) 

        if timestamp is not None:
            return timestamp
        else:
            print(text)
            text = self.ocr.readtext(bottom_image)
            if len(text):
                text= text[0][1]
                timestamp = self.chinese_date_to_timestamp(text) 
            else:
                timestamp = None
            if timestamp is not None:
                return timestamp
            else:
                print(text)
                # ocr failed:
                result = self.paddleocr.predict(bottom_image)
                text = "".join(result[0]['rec_texts'])
                timestamp = self.chinese_date_to_timestamp(text) 
                if timestamp:
                    return timestamp
                else:
                    # ocr failed:
                    result = self.paddleocr.predict(original_image)
                    text = "".join(result[0]['rec_texts'])
                    timestamp = self.chinese_date_to_timestamp(text) 
                    if timestamp:
                        return timestamp
                    else:
                        print('fail to ocr')
                        fn = str(self.debug_dir / f'{self.current_moment_index}.png')
                        cv2.imwrite(fn, original_image)
                        return fn

    def chinese_date_to_timestamp(self, date_str):
        """
        将中文日期时间字符串转换为时间戳
        格式：xxx年x月x日 xx:xx
        """
        # 使用正则表达式提取日期时间组件
        pattern = r'(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})(:|：)(\d{1,2})'
        match = re.search(pattern, date_str.strip().replace(' ',''))
        
        if not match:
            return None
        
        year, month, day, hour, _, minute = match.groups()
        
        # 创建datetime对象
        dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
        
        # 转换为时间戳
        timestamp = dt.isoformat()
        
        return timestamp
     

    def save_posts(self, moment_index: int) -> List[str]:
        if not self.text_position:
            # 第一次，让用户点击并记录
            self.record_text_postion()
        text = self.select_and_get_text()
        if not text:
            pyautogui.press('esc')
            time.sleep(2)
        # print(f'get text: {text}')
        record_id = text[:30]
        if record_id != '' and record_id in self.record_keys:
            print('have record')
            return None, None, None
    

        saved_images, time_crop, original_image = self.save_images_auto(moment_index=moment_index)
        timestamp = self.get_post_time(time_crop, original_image)
        return text, saved_images, timestamp

    def find_blocks_by_background_subtraction(self, screenshot_cv,
                                            color_tolerance=4,
                                            min_area=2000,
                                            min_rectangularity=0.95,
                                            aspect_ratio_range=(0.3, 2.5)):
        """
        通过减去背景颜色来检测方块轮廓并计算中心点
        
        参数:
            background_color: 背景颜色 (R, G, B)
            color_tolerance: 颜色容差范围
            min_area: 最小轮廓面积
            max_area: 最大轮廓面积
            aspect_ratio_range: 宽高比范围 (min, max)
        
        返回:
            blocks: 包含方块信息的列表
        """
        # 3. 创建背景颜色掩码
        # 将背景颜色转换为numpy数组 (注意OpenCV使用BGR格式)
        bg_color_bgr = np.array([self.background_color, self.background_color, self.background_color])
        
        # 计算颜色范围
        lower_bound = np.array([
            max(0, bg_color_bgr[0] - color_tolerance),
            max(0, bg_color_bgr[1] - color_tolerance),
            max(0, bg_color_bgr[2] - color_tolerance)
        ])
        
        upper_bound = np.array([
            min(255, bg_color_bgr[0] + color_tolerance),
            min(255, bg_color_bgr[1] + color_tolerance),
            min(255, bg_color_bgr[2] + color_tolerance)
        ])
        
        # 创建掩码：背景区域为0，非背景区域为255
        mask = cv2.inRange(screenshot_cv, lower_bound, upper_bound)
        
        # 反转掩码，使背景为0，前景为255
        foreground_mask = cv2.bitwise_not(mask)

        # 1️⃣ 提取水平线
        # h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        # h_lines = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, h_kernel)
        # 5. 查找轮廓
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       
        min_score = 100000
        bottom_contour = None

        blocks = []
        res = []
        for i, contour in enumerate(contours):
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 获取边界矩形
            x, y, w, h = cv2.boundingRect(contour)

            if w > 400:
                continue

            score = abs(x - 482) + abs(y - 675) + abs(w-32) + abs(h-20)
            # (482, 675, 32, 20)
            if min_score > score:
                bottom_contour = (x, y, w, h)
                min_score = score
        
            # 过滤面积过小或过大的轮廓
            if area < min_area:
                # print(i, 'area', area)
                continue
            
            # 计算宽高比
            aspect_ratio = w / h if h > 0 else 0
            
            # 过滤不符合宽高比范围的轮廓
            if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
                # print(i, 'aspect_ratio', aspect_ratio)
                continue
            
            # 计算矩形度 = 轮廓面积 / 边界矩形面积
            rect_area = w * h

            rectangularity = area / rect_area if rect_area > 0 else 0
        
            if rectangularity < min_rectangularity:
                # print(i, 'rectangularity',rectangularity)
                continue

            # 计算中心点
            center_x = x + w // 2
            center_y = y + h // 2

            
            blocks.append({
                'contour': contour,
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'area': area,
                'aspect_ratio': aspect_ratio
            })
            res.append((center_x, center_y))
        cropped_image = screenshot_cv[bottom_contour[1]:bottom_contour[1]+bottom_contour[3], :]
        return res, cropped_image

    def sort_grid_positions(self, positions, y_tolerance=5):
        """
        排序2D坐标：整体从上往下，每行从左往右
        y_tolerance: Y坐标容差，小于此值视为同一行
        """
        # 按Y坐标分组
        sorted_by_y = sorted(positions, key=lambda p: p[1])
        
        rows = []
        current_row = [sorted_by_y[0]]
        current_y = sorted_by_y[0][1]
        
        for point in sorted_by_y[1:]:
            if abs(point[1] - current_y) <= y_tolerance:
                current_row.append(point)
            else:
                # 对当前行按X坐标排序
                rows.append(sorted(current_row, key=lambda p: p[0]))
                current_row = [point]
                current_y = point[1]
        
        # 处理最后一行
        if current_row:
            rows.append(sorted(current_row, key=lambda p: p[0]))
        
        # 展平结果
        return [point for row in rows for point in row]

    def save_images_auto(self, moment_index: int) -> List[str]:
        """自动检测并保存图片"""
        pyautogui.press('up')
        time.sleep(self.config["delays"]["scroll"])  
        region_screenshot = pyautogui.screenshot(region=self.region)
        region_np = np.array(region_screenshot)
        region_cv = cv2.cvtColor(region_np, cv2.COLOR_RGB2BGR)

        # 检测图片位置
        image_positions, cropped_image = self.find_blocks_by_background_subtraction(region_cv)
        image_positions.sort(key=lambda p: p[1]+p[0])
        image_positions = self.sort_grid_positions(image_positions)

        print(f"  检测到 {len(image_positions)} 个图片区域")

        # 保存图片

        saved_images = []
        count = len(image_positions)

        for i in range(count):
            print(f"  保存 {i+1}/{count}...")
            # 截图
            filename = f"{moment_index:04d}_{i+1:02d}"
            filepath = self.images_dir / filename
            
            if i ==0:
                x, y = image_positions[i]
                x = x + self.region[0]
                y = y + self.region[1]
                # 点击图片
                pyautogui.click(x, y)
                time.sleep(self.config["delays"]["page_load"])

            # 输入文件名
            pyautogui.hotkey('command', 's')
            time.sleep(self.config["delays"]["hotkey"])

            pyautogui.write(filename)
            time.sleep(self.config["delays"]["click"])
            
            # 按回车保存
            pyautogui.press('enter')
            time.sleep(self.config["delays"]["click"])

            saved_images.append(str(filepath))

            # 右边
            if i < count-1:
                pyautogui.press('right')
                time.sleep(self.config["delays"]["between_images"])
        pyautogui.press('esc')

        # check save results
        for i, fn in enumerate(saved_images):
            if not glob.glob(fn+'*'):
                filename = f"{moment_index:04d}_{i+1:02d}"
                filepath = self.images_dir / filename
                x, y = image_positions[i]
                x = x + self.region[0]
                y = y + self.region[1]
                flag = False
                for j in range(20):
                    # 点击图片
                    pyautogui.click(x, y)
                    time.sleep(1.0)

                    # 输入文件名
                    pyautogui.hotkey('command', 's')
                    time.sleep(1.0)

                    pyautogui.write(filename)
                    time.sleep(self.config["delays"]["click"])
                    
                    # 按回车保存
                    pyautogui.press('enter')
                    time.sleep(self.config["delays"]["click"])

                    pyautogui.press('esc')
                    if glob.glob(fn+'*'):
                        flag = True
                        break
                if not flag:
                    # can't save the picture
                    raise(f"can't save the picture {filename}")


        print(f"✓ 成功保存 {len(saved_images)} 张")
        return saved_images, cropped_image, region_cv

    def record_text_postion(self):
        """记录text start位置"""
        print("\n" + "="*60)
        print("步骤: 记录text start位置（仅需一次）")
        print("="*60)
        print("请将光标指向\"第一个字符\"")
        print("然后按Enter，我会记录这个位置，之后就自动点击")
        print("="*60)

        # 等待用户点击
        print("\n等待你将光标移到首字符...")

        # 监控鼠标点击
        # 简化方案：让用户点击后，手动标记位置
        input("\n点击完IDE/terminal后，移动鼠标到首字符上按Enter: ")
        time.sleep(self.config["delays"]["click"])

        self.text_position = pyautogui.position()
        print(f"✓ 已记录首字符位置: {self.text_position}")
        print("  之后将自动点击此位置记录文本\n")

    def record_back_button(self):
        """记录返回按钮位置"""
        print("\n" + "="*60)
        print("步骤: 记录返回按钮位置（仅需一次）")
        print("="*60)
        print("请将光标移动到返回按钮（通常是左上角的 ← 或 ×）")
        print("然后按Enter，我会记录这个位置，之后就自动返回")
        print("="*60)

        # 等待用户点击
        print("\n等待你将光标移到返回按钮...")

        # 监控鼠标点击
        # 简化方案：让用户点击后，手动标记位置
        input("\n点击IDE/terminal后，移动鼠标到按钮上按Enter: ")
        time.sleep(self.config["delays"]["click"])

        self.back_button_position = pyautogui.position()
        print(f"✓ 已记录返回按钮位置: {self.back_button_position}")
        print("  之后将自动点击此位置返回\n")
        pyautogui.click(self.back_button_position[0], self.back_button_position[1])

    def go_back(self):
        """返回主界面"""
        if self.back_button_position:
            # 自动点击返回
            x, y = self.back_button_position
            print(f"  点击返回按钮: ({x}, {y})")
            pyautogui.click(x, y)
        else:
            # 第一次，让用户点击并记录
            self.record_back_button()

        time.sleep(self.config["delays"]["back"])


    def bottom_lines_changed(self, template, curr_img, match_thresh=0.98):
        """
        判断上一屏底部 num_lines 行是否在当前屏幕中消失
        match_thresh: 模板匹配相似度阈值
        """
        res = cv2.matchTemplate(curr_img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        print(max_val, match_thresh)
        if max_val > match_thresh and max_loc[1] < 60:
            # 上一页最后post移到当前前100行了，就可以停止，不让很容易滑过界
            return True
        else:
            # 如果最高匹配度 < 阈值，则认为底部不在屏幕中
            return max_val < match_thresh


    def scroll_to_newpage(self, prev_img):
        last_img = None
        for i in range(self.config["scroll"]["max_scrolls"]):
            pyautogui.scroll(self.config["scroll"]["amount"])
            time.sleep(self.config["delays"]["scroll"])
            screenshot = pyautogui.screenshot(region=self.region)
            curr_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
            # import pdb;pdb.set_trace()
            
            if last_img is not None and np.sum(curr_img -last_img) < 10:
                # to bottom
                print('滑不动了')
                return True
            if self.bottom_lines_changed(prev_img, curr_img):
                print("新区域到达！")
                break
            else:
                print("还在旧区域，继续滚动...")
                last_img = curr_img
        return False

    def scroll_page(self):
        """滚动页面"""
        if self.region:
            left, top, width, height = self.region
            center_x = left + width // 2
            center_y = top + height // 2

            pyautogui.moveTo(center_x, center_y)
            time.sleep(0.2)

            finished = self.scroll_to_newpage(self.curview)
            self.scroll_count += 1
            print(f"  已滚动 {self.scroll_count} 次")
        return finished

    def restore_scroll_page(self):
        """滚动页面"""
        if self.region:
            left, top, width, height = self.region
            center_x = left + width // 2
            center_y = top + height // 2

            pyautogui.moveTo(center_x, center_y)
            time.sleep(0.2)
            count_post = 0
            for i in range(self.scroll_count):
                post, self.curview = self.find_post_area(self.region)
                finished = self.scroll_to_newpage(self.curview)
                count_post += len(post)
            print(f"  已滚动 {self.scroll_count} 次")
        return count_post

    def save_checkpoint(self):
        """保存检查点"""
        try:
            data = {
                "records": [asdict(r) for r in self.records],
                "current_moment_index": self.current_moment_index,
                "scroll_time": self.scroll_count,
                "last_updated": datetime.now().isoformat(), "text_position": self.text_position,
                "region":self.region,
                "record_keys": self.record_keys,
                "back_button": self.back_button_position,
                "center_x": self.center_x,
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  保存检查点失败: {e}")

    def load_checkpoint(self):
        """加载检查点"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.records = [MomentRecord(**r) for r in data.get("records", [])]
                    self.current_moment_index = data.get("current_moment_index", 0)
                    self.scroll_count = data.get("scroll_time", 0)
                    self.center_x = data.get('center_x', 0)
                    self.region = data.get('region', None)
                    self.record_keys = data.get('record_keys', [])
                    self.text_position = data.get('text_position', None)
                    self.back_button_position = data.get('back_button', None)
                    print(f"✓ 已加载检查点: {len(self.records)} 条")
            except Exception as e:
                print(f"⚠️  加载检查点失败: {e}")

    def image_hash_md5(self, img):
        # 缩小尺寸可减少敏感度
        small = cv2.resize(img, (64, 64))
        # 转灰度，转 bytes
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        data = gray.tobytes()
        return hashlib.md5(data).hexdigest()

    def run(self):
        """主循环"""
        print("\n" + "="*60)
        print("微信朋友圈智能爬虫")
        print("="*60)
        print("\n⚠️  注意:")
        print("1. 确保微信窗口可见")
        print("2. Ctrl+C可随时中断")
        print("="*60 + "\n")

        try:
            # 加载
            if self.state == State.INIT:
                self.state = State.SELECT_REGION

            input("\n按 Enter 开始...")

            # 选择区域
            if self.state == State.SELECT_REGION:
                if not self.region:
                    self.region = self.select_region()
                    self.center_x = self.region[0] + self.region[2] // 2
                self.state = State.SCAN
                # 开始想做scroll的restore，因为scroll其实不准，所以干脆不用了
                # if self.scroll_count > 0:
                #     past_post_len = self.restore_scroll_page()
                #     finished_post_len = self.current_moment_index - past_post_len
                # else:
                #     finished_post_len = self.current_moment_index
                finished_post_len=0
                self.save_checkpoint()

            # 主循环
            while self.state != State.DONE:
                if self.state == State.SCAN:
                    # post area
                    print(f"\n[{self.current_moment_index}] 扫描屏幕...")
                    self.posts, self.curview = self.find_post_area(self.region)
                    if finished_post_len > 0:
                        self.posts = self.posts[finished_post_len:]
                        finished_post_len = 0
                    print(f'find {len(self.posts)} posts in this page')
                    self.state = State.MATCH_TEXT
                elif self.state == State.MATCH_TEXT:
                    cur = self.posts.pop(0)
                    x = self.center_x
                    y = self.region[1] + (cur[0] + cur[1])//2
                    pyautogui.click(x, y)
                    time.sleep(self.config["delays"]["click"])
                    self.state = State.WAIT_DETAIL
                elif self.state == State.SCROLL:
                    finished = self.scroll_page()
                    if finished:
                        self.state = State.DONE
                    else:
                        self.state = State.SCAN
                elif self.state == State.WAIT_DETAIL:
                    # 等待详情页
                    print("  等待详情页加载...")
                    time.sleep(self.config["delays"]["page_load"])
                    
                    region_screenshot = pyautogui.screenshot(region=self.region)
                    curr_img = cv2.cvtColor(np.array(region_screenshot), cv2.COLOR_RGB2BGR)
                    # import pdb;pdb.set_trace()
                    # self.state = State.SAVE_IMAGES
                    if self.bottom_lines_changed(self.curview, curr_img):
                        # 已跳转
                        self.state = State.SAVE_IMAGES
                    elif len(self.posts):
                        self.state = State.MATCH_TEXT
                    else:
                        self.state = State.SCROLL
                elif self.state == State.SAVE_IMAGES:
                    # 3. 截取该区域图像
                    # region_screenshot = pyautogui.screenshot(region=self.region)
                    # region_np = np.array(region_screenshot)
                    # record_id = self.image_hash_md5(region_np)
                    text, images, timestamp = self.save_posts(
                        self.current_moment_index)

                    if text is not None:    
                        # 创建记录
                        record = MomentRecord(
                            # id=record_id,
                            moment_index=self.current_moment_index,
                            timestamp=timestamp,
                            text=text,
                            images=images,
                        )

                        if text != "":
                            self.record_keys.append(text[:30])
                        self.records.append(record)
                        self.current_moment_index += 1
                        print(f"\n✓ 完成第 {len(self.records)} 条")
                        self.save_checkpoint()

                    self.state = State.GO_BACK

                elif self.state == State.GO_BACK:
                    # 返回
                    self.go_back()
                    if len(self.posts):
                        self.state = State.MATCH_TEXT
                    else:
                        self.state = State.SCROLL

            print("\n" + "="*60)
            print("抓取完成！")
            print("="*60)
            print(f"✓ 总共抓取: {len(self.records)} 条")
            print(f"✓ 保存位置: {self.save_dir}")
            print("="*60 + "\n")

        except KeyboardInterrupt:
            print("\n\n⚠️  程序中断")
            self.save_checkpoint()
            print("✓ 进度已保存")

        except Exception as e:
            print(f"\n❌ 程序出错: {e}")
            import traceback
            traceback.print_exc()
            self.save_checkpoint()



def main():
    """主函数"""
    color = 46
    if len(sys.argv) > 1:
        color = int(sys.argv[1])
    crawler = WeChatSmartCrawler(background_color=color)
    crawler.run()


if __name__ == "__main__":
    main()
