import cv2
import numpy as np
import os
import sys
from typing import Tuple, List, Optional

# 可选：在 Windows 控制台启用 UTF-8，避免中文打印为乱码
def _configure_console_encoding():
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if os.name == "nt":
            # 将控制台输出代码页设置为 UTF-8（65001）
            import ctypes
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

_configure_console_encoding()

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# 可选：MediaPipe 人脸关键点（用 importlib 动态加载，避免静态导入告警）
import importlib
MP_AVAILABLE = False
MP_IMPORT_ERROR = None
try:
    mp = importlib.import_module("mediapipe")
    MP_AVAILABLE = True
except Exception as _e:
    MP_IMPORT_ERROR = str(_e)


class TextRenderer:
    """使用 Pillow 在 OpenCV 图像上绘制中文/英文文本，避免 cv2.putText 的中文乱码问题。"""
    def __init__(self):
        self.available = False
        if not PIL_AVAILABLE:
            return
        # 常见中文字体候选（Windows）
        candidates = [
            r"C:\\Windows\\Fonts\\msyh.ttc",      # 微软雅黑
            r"C:\\Windows\\Fonts\\msyhl.ttc",     # 微软雅黑 Light
            r"C:\\Windows\\Fonts\\simhei.ttf",    # 黑体
            r"C:\\Windows\\Fonts\\simsun.ttc",    # 宋体
            r"C:\\Windows\\Fonts\\simkai.ttf",    # 楷体
        ]
        font_path = None
        for p in candidates:
            if os.path.exists(p):
                font_path = p
                break
        try:
            self.font_path = font_path
            self.available = True if font_path else False
        except Exception:
            self.available = False

    def draw_text(self, frame, text: str, pos: Tuple[int, int], font_size: int = 24,
                  color: Tuple[int, int, int] = (255, 255, 255),
                  bg: Tuple[int, int, int] | None = None):
        """在 BGR 图像上绘制文本。color/bg 都是 BGR 颜色。"""
        if not self.available:
            return frame
        try:
            # OpenCV BGR -> PIL RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype(self.font_path, font_size)

            # 文字背景（可选）
            if bg is not None:
                # 先估算文本尺寸
                text_bbox = draw.textbbox(pos, text, font=font)
                # BGR->RGB
                bg_rgb = (bg[2], bg[1], bg[0])
                draw.rectangle(text_bbox, fill=bg_rgb)

            # 字体颜色 BGR->RGB
            rgb = (color[2], color[1], color[0])
            draw.text(pos, text, font=font, fill=rgb)

            # 回写到 OpenCV BGR
            out = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
            return out
        except Exception:
            # 任何异常回退原图
            return frame


class MaskAnchors:
    def __init__(self, left_eye: Tuple[int,int], right_eye: Tuple[int,int], chin: Tuple[int,int]):
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.chin = chin


class MaskAsset:
    def __init__(self, name: str, image: np.ndarray, anchors: "MaskAnchors"):
        self.name = name
        self.image = image  # RGBA
        self.anchors = anchors

class FaceMaskChanger:
    def __init__(self):
        # 加载人脸检测器
        cascade_file = 'haarcascade_frontalface_default.xml'
        default_cascade_path = cv2.data.haarcascades + cascade_file
        self.face_cascade = cv2.CascadeClassifier(default_cascade_path)
        
        # 若因路径含有特殊字符导致加载失败，则复制到本地 ASCII 路径再加载
        if self.face_cascade.empty():
            try:
                local_cascade_path = os.path.join(os.path.dirname(__file__), cascade_file)
                if not os.path.exists(local_cascade_path):
                    with open(default_cascade_path, 'rb') as src, open(local_cascade_path, 'wb') as dst:
                        dst.write(src.read())
                self.face_cascade = cv2.CascadeClassifier(local_cascade_path)
            except Exception:
                pass
        
        if self.face_cascade.empty():
            raise RuntimeError(
                "人脸检测器加载失败。请确认 OpenCV 安装正常，且已包含 haarcascade 数据文件（已尝试本地复制仍失败）。"
            )
        
        # 面具列表：优先加载 masks 目录 PNG；无则生成带形态的椭圆面具
        self.masks = []
        self.load_masks()
        
        self.current_mask_index = 0
        self.face_detected = False
        self.face_lost_frames = 0
        self.face_lost_threshold = 10  # 连续10帧检测不到人脸才认为脸被挡住了
        self.lost_notice_printed = False  # 控制“准备切换面具”提示仅打印一次
        self.text_renderer = TextRenderer() if PIL_AVAILABLE else None
        # 开关：是否在切换到“随机面具”槽位时重新生成（保证每次都不一样）
        self.randomize_on_switch = True
        
    def load_masks(self):
        """加载面具图像：
        1) 尝试从 ./masks/*.png 加载带透明通道的 PNG；
        2) 若没有，则程序化生成“脸型”椭圆面具（带眼洞与装饰）。
        """
        # 若 masks 目录不存在，先生成一套默认 PNG 到磁盘，便于用户查看和替换
        masks_dir = os.path.join(os.path.dirname(__file__), 'masks')
        if not os.path.isdir(masks_dir):
            try:
                os.makedirs(masks_dir, exist_ok=True)
                self._generate_default_png_masks_to_disk(masks_dir)
            except Exception:
                pass

        loaded = self._load_external_png_masks()
        if loaded:
            # 若外部 PNG 存在，仍然在最前面插入一个“随机面具”槽位，方便体验随机款
            self.masks = loaded
            try:
                self.masks.insert(0, self.create_shape_mask_random(name="random_auto"))
            except Exception:
                pass
            return

        # 生成更丰富的配色主题（更复杂的川剧风格）
        base_list = [
            self.create_shape_mask((40, 40, 240),   (255, 255, 255), name="opera_red"),     # 红 + 白描
            self.create_shape_mask((255, 80, 30),   (255, 255, 0),   name="opera_blue"),    # 蓝 + 金黄
            self.create_shape_mask((30, 200, 30),   (0, 0, 0),       name="opera_green"),   # 绿 + 黑描
            self.create_shape_mask((240, 240, 240), (0, 0, 0),       name="opera_white"),   # 白 + 黑描
            self.create_shape_mask((20, 20, 20),    (255, 255, 255), name="opera_black"),   # 黑 + 白描
            self.create_shape_mask((160, 170, 10),  (255, 215, 0),   name="opera_gold"),    # 金绿 + 金描
            self.create_shape_mask((180, 40, 180),  (255, 255, 255), name="opera_purple"),  # 紫 + 白描
            self.create_shape_mask((240, 190, 40),  (0, 0, 0),       name="opera_orange"),  # 橙 + 黑描
        ]
        # 在第 1 个位置放入“随机面具”槽位
        try:
            self.masks = [self.create_shape_mask_random(name="random_auto")] + base_list
        except Exception:
            self.masks = base_list

    def create_shape_mask_random(self, name: str = "random_auto") -> "MaskAsset":
        """基于 create_shape_mask 生成随机款：颜色/描边/装饰参数都带随机性，确保每次都不一样。"""
        # 随机挑选高对比配色或生成互补色
        palette = [
            ((40, 40, 240),   (255, 255, 255)),
            ((255, 80, 30),   (255, 255, 0)),
            ((30, 200, 30),   (0, 0, 0)),
            ((240, 240, 240), (0, 0, 0)),
            ((20, 20, 20),    (255, 255, 255)),
            ((160, 170, 10),  (255, 215, 0)),
            ((180, 40, 180),  (255, 255, 255)),
            ((240, 190, 40),  (0, 0, 0)),
        ]
        if np.random.rand() < 0.5:
            base_color_bgr, outline_bgr = palette[int(np.random.rand()*len(palette))]
        else:
            # 随机色 + 自动对比描边
            base_color_bgr = tuple(int(x) for x in np.random.randint(20, 236, size=3))
            luminance = 0.114*base_color_bgr[0] + 0.587*base_color_bgr[1] + 0.299*base_color_bgr[2]
            outline_bgr = (255,255,255) if luminance < 128 else (0,0,0)

        # 随机扰动：眉弓/旋纹数量、粗细、梯度强度
        # 通过设置随机状态，create_shape_mask 内部使用这些参数的默认推导
        # 这里简单调用原函数，它已包含较多元素；我们通过颜色变化 + 后续小抖动来制造差异
        asset = self.create_shape_mask(base_color_bgr, outline_bgr, name=name)

        # 小规模“喷溅/点缀”，增强独特性（对 RGB 通道直接绘制，不改 alpha）
        H, W = asset.image.shape[:2]
        dots = np.random.randint(6, 14)
        for _ in range(dots):
            cx = int(np.random.uniform(W*0.2, W*0.8))
            cy = int(np.random.uniform(H*0.15, H*0.85))
            r  = int(np.random.uniform(max(2, W*0.008), max(3, W*0.02)))
            col = tuple(int(c) for c in np.clip(np.array(outline_bgr) + np.random.randint(-30, 31, size=3), 0, 255))
            cv2.circle(asset.image, (cx, cy), r, col, -1)

        # 轻微随机“裂纹”线条
        lines = np.random.randint(2, 5)
        for _ in range(lines):
            x1 = int(np.random.uniform(W*0.25, W*0.75))
            y1 = int(np.random.uniform(H*0.15, H*0.35))
            x2 = x1 + int(np.random.uniform(-W*0.15, W*0.15))
            y2 = y1 + int(np.random.uniform(H*0.10, H*0.30))
            thickness = int(np.random.uniform(1, 4))
            cv2.line(asset.image, (x1,y1), (x2,y2), outline_bgr, thickness, lineType=cv2.LINE_AA)

        # 名称附带时间戳后缀用于区分
        try:
            import time
            asset.name = f"{name}_{int(time.time()*1000)%100000}"
        except Exception:
            pass
        return asset

    def _maybe_regenerate_random_mask(self):
        """若当前选择的是随机槽位且开启开关，则重新生成一个随机面具。"""
        try:
            if not self.randomize_on_switch:
                return
            if 0 <= self.current_mask_index < len(self.masks):
                cur = self.masks[self.current_mask_index]
                if isinstance(cur, MaskAsset) and isinstance(cur.name, str) and cur.name.startswith("random"):
                    self.masks[self.current_mask_index] = self.create_shape_mask_random(name="random_auto")
        except Exception:
            pass
        
    def _load_external_png_masks(self) -> List["MaskAsset"]:
        """从 ./masks 目录加载 PNG 面具（需带透明通道）。"""
        masks_dir = os.path.join(os.path.dirname(__file__), 'masks')
        if not os.path.isdir(masks_dir):
            return []
        masks: List[MaskAsset] = []
        for name in os.listdir(masks_dir):
            if not name.lower().endswith('.png'):
                continue
            path = os.path.join(masks_dir, name)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            # 统一转 RGBA
            if img.shape[2] == 3:
                # 没有 alpha 的 PNG，添加全不透明 alpha
                b,g,r = cv2.split(img)
                a = np.full_like(b, 255)
                img = cv2.merge((b,g,r,a))
            H, W = img.shape[:2]
            # 对外部 PNG，使用启发式锚点（左右眼外侧 + 下巴）
            anchors = MaskAnchors(
                left_eye=(int(W*0.32), int(H*0.40)),
                right_eye=(int(W*0.68), int(H*0.40)),
                chin=(int(W*0.50), int(H*0.86)),
            )
            masks.append(MaskAsset(name=os.path.splitext(name)[0], image=img, anchors=anchors))
        return masks

    def create_shape_mask(self, base_color_bgr, outline_bgr, name: str = "generated") -> "MaskAsset":
        """生成更复杂的“脸型”川剧风格面具（RGBA）。
        - 基础脸型 + 渐变明暗
        - 眼洞 + 眼周描边、眉弓
        - 额头徽记（多层图形）
        - 面颊旋纹、鼻梁点缀、嘴部纹样
        """
        W, H = 420, 520
        rgba = np.zeros((H, W, 4), dtype=np.uint8)

        # 基础脸廓 alpha
        alpha = np.zeros((H, W), dtype=np.uint8)
        center = (W//2, int(H*0.48))
        axes = (int(W*0.42), int(H*0.46))
        cv2.ellipse(alpha, center, axes, 0, 0, 360, 235, -1)

        # 眼洞
        eye_y = int(H*0.42)
        eye_dx = int(W*0.165)
        eye_r = int(W*0.07)
        cv2.circle(alpha, (center[0]-eye_dx, eye_y), eye_r, 0, -1)
        cv2.circle(alpha, (center[0]+eye_dx, eye_y), eye_r, 0, -1)

        # 底色
        mask_face = alpha > 0
        rgba[mask_face, 0] = base_color_bgr[0]
        rgba[mask_face, 1] = base_color_bgr[1]
        rgba[mask_face, 2] = base_color_bgr[2]
        rgba[..., 3] = alpha

        # 渐变明暗（上亮下暗）
        yy, xx = np.mgrid[0:H, 0:W]
        y_norm = yy.astype(np.float32) / max(1, H)
        shade = (1.10 - 0.25 * y_norm).clip(0.7, 1.10)
        for c in range(3):
            ch = rgba[..., c].astype(np.float32)
            ch[mask_face] = (ch[mask_face] * shade[mask_face]).clip(0, 255)
            rgba[..., c] = ch.astype(np.uint8)

        # 外轮廓描边
        outline_img = np.zeros((H, W, 3), dtype=np.uint8)
        cv2.ellipse(outline_img, center, axes, 0, 0, 360, outline_bgr, 7)
        rgba[..., :3] = np.clip(rgba[..., :3].astype(np.int32) + outline_img.astype(np.int32), 0, 255).astype(np.uint8)

        # 眼周双层描边 + 高光
        accent1 = (255, 255, 255) if outline_bgr != (255, 255, 255) else (0, 0, 0)
        accent2 = (0, 0, 0) if accent1 == (255, 255, 255) else (255, 255, 255)
        for dx in (-eye_dx, eye_dx):
            cv2.circle(rgba, (center[0] + dx, eye_y), int(eye_r * 1.25), accent1, 3)
            cv2.circle(rgba, (center[0] + dx, eye_y), int(eye_r * 1.45), accent2, 2)
            cv2.circle(rgba, (center[0] + dx - int(eye_r*0.4), eye_y - int(eye_r*0.3)), int(max(1, eye_r*0.12)), (255,255,255), -1)

        # 眉弓
        brow_y = eye_y - int(H*0.06)
        brow_w = int(W*0.11)
        brow_h = int(H*0.05)
        for sign in (-1, 1):
            bx = center[0] + sign * eye_dx
            cv2.ellipse(rgba, (bx, brow_y), (brow_w, brow_h), 0, 200 if sign<0 else -20, 20 if sign<0 else 200, outline_bgr, 3)

        # 额头徽记（菱形 + 火焰三角）
        fw_y = int(H*0.20)
        size = int(W*0.06)
        pts = np.array([
            (center[0], fw_y - size),
            (center[0] + size, fw_y),
            (center[0], fw_y + size),
            (center[0] - size, fw_y),
        ], dtype=np.int32)
        cv2.fillConvexPoly(rgba, pts, accent1)
        cv2.polylines(rgba, [pts], True, accent2, 2)
        tri = np.array([
            (center[0], fw_y - size - int(size*0.9)),
            (center[0] + int(size*0.55), fw_y - int(size*0.1)),
            (center[0] - int(size*0.55), fw_y - int(size*0.1)),
        ], dtype=np.int32)
        cv2.fillConvexPoly(rgba, tri, outline_bgr)

        # 面颊旋纹
        cheek_y = int(H*0.60)
        swirl_r = int(W*0.055)
        for sign in (-1, 1):
            cx = center[0] + sign * int(W*0.18)
            cv2.ellipse(rgba, (cx, cheek_y), (swirl_r, int(swirl_r*0.7)), 0, 30 if sign<0 else 150, 220 if sign<0 else 340, accent1, 3)
            cv2.ellipse(rgba, (cx, cheek_y), (int(swirl_r*0.7), int(swirl_r*0.45)), 0, 30 if sign<0 else 150, 220 if sign<0 else 340, accent2, 2)

        # 鼻梁 + 嘴部
        cv2.circle(rgba, (center[0], int(H*0.50)), int(W*0.018), accent1, -1)
        mouth_y = int(H*0.70)
        mouth_w = int(W*0.20)
        mouth_h = int(H*0.06)
        cv2.ellipse(rgba, (center[0], mouth_y), (mouth_w, mouth_h), 0, 200, 340, outline_bgr, 4)
        cv2.ellipse(rgba, (center[0], mouth_y+int(H*0.01)), (int(mouth_w*0.6), int(mouth_h*0.45)), 0, 200, 340, accent2, 2)

        anchors = MaskAnchors(
            left_eye=(center[0]-eye_dx, eye_y),
            right_eye=(center[0]+eye_dx, eye_y),
            chin=(center[0], int(H*0.90)),
        )
        return MaskAsset(name=name, image=rgba, anchors=anchors)

    def _generate_default_png_masks_to_disk(self, out_dir: str):
        """将程序化面具导出为 PNG 文件，便于用户查看与替换。"""
        assets = [
            self.create_shape_mask((40, 40, 240),   (255, 255, 255), name="opera_red"),
            self.create_shape_mask((255, 80, 30),   (255, 255, 0),   name="opera_blue"),
            self.create_shape_mask((30, 200, 30),   (0, 0, 0),       name="opera_green"),
            self.create_shape_mask((240, 240, 240), (0, 0, 0),       name="opera_white"),
            self.create_shape_mask((20, 20, 20),    (255, 255, 255), name="opera_black"),
            self.create_shape_mask((160, 170, 10),  (255, 215, 0),   name="opera_gold"),
            self.create_shape_mask((180, 40, 180),  (255, 255, 255), name="opera_purple"),
            self.create_shape_mask((240, 190, 40),  (0, 0, 0),       name="opera_orange"),
        ]
        for a in assets:
            path = os.path.join(out_dir, f"{a.name}.png")
            cv2.imwrite(path, a.image)
        
    def apply_mask_rect(self, frame, face):
        """矩形贴合（无关键点时的回退方案）。face: (x,y,w,h)"""
        x, y, w, h = face
        H, W = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return frame
        w2, h2 = x1 - x0, y1 - y0
        asset = self.masks[self.current_mask_index]
        mask_resized = cv2.resize(asset.image, (w2, h2), interpolation=cv2.INTER_LINEAR)
        mask_rgb = mask_resized[:, :, :3].astype(np.float32)
        mask_alpha = (mask_resized[:, :, 3].astype(np.float32) / 255.0)
        mask_alpha_3 = cv2.merge([mask_alpha, mask_alpha, mask_alpha])
        face_region = frame[y0:y1, x0:x1].astype(np.float32)
        blended = face_region * (1.0 - mask_alpha_3) + mask_rgb * mask_alpha_3
        frame[y0:y1, x0:x1] = blended.astype(np.uint8)
        return frame

    def apply_mask_landmarks(self, frame, landmarks: List[Tuple[int,int]]):
        """使用三点仿射（左右眼外侧+下巴）将面具变形贴合到脸部。"""
        if len(landmarks) == 0:
            return frame
        # 取 MediaPipe 的三个关键点：左眼外 263，右眼外 33，下巴 152
        try:
            ptL = landmarks[263]
            ptR = landmarks[33]
            ptC = landmarks[152]
        except Exception:
            return frame
        asset = self.masks[self.current_mask_index]
        src = np.float32([
            [asset.anchors.left_eye[0], asset.anchors.left_eye[1]],
            [asset.anchors.right_eye[0], asset.anchors.right_eye[1]],
            [asset.anchors.chin[0], asset.anchors.chin[1]],
        ])
        dst = np.float32([
            [ptL[0], ptL[1]],
            [ptR[0], ptR[1]],
            [ptC[0], ptC[1]],
        ])
        M = cv2.getAffineTransform(src, dst)
        H, W = frame.shape[:2]
        warped = cv2.warpAffine(asset.image, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
        mask_rgb = warped[:, :, :3].astype(np.float32)
        mask_alpha = (warped[:, :, 3].astype(np.float32) / 255.0)
        mask_alpha_3 = cv2.merge([mask_alpha, mask_alpha, mask_alpha])
        base = frame.astype(np.float32)
        blended = base * (1.0 - mask_alpha_3) + mask_rgb * mask_alpha_3
        frame[:, :, :] = blended.astype(np.uint8)
        return frame
    
    def process_frame(self, frame):
        """处理视频帧"""
        # 首选 MediaPipe FaceMesh（如可用），否则回退到 Haar
        faces = []
        used_landmarks = False
        if MP_AVAILABLE:
            if not hasattr(self, "_mp_face"):
                self._mp_face = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    refine_landmarks=True,
                    max_num_faces=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self._mp_face.process(rgb)
            if res.multi_face_landmarks:
                used_landmarks = True
                # 将 468 个点映射到像素坐标列表
                H, W = frame.shape[:2]
                lms = res.multi_face_landmarks[0]
                pts = [(int(p.x * W), int(p.y * H)) for p in lms.landmark]
                frame = self.apply_mask_landmarks(frame, pts)
                faces = [cv2.boundingRect(np.array(pts))]
        if not used_landmarks:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

        if len(faces) > 0:
            # 检测到人脸
            if not self.face_detected:
                self.face_detected = True
                self.face_lost_frames = 0
                self.lost_notice_printed = False
            else:
                # 如果之前脸被挡住，现在重新检测到，就切换面具
                if self.face_lost_frames >= self.face_lost_threshold:
                    self.current_mask_index = (self.current_mask_index + 1) % len(self.masks)
                    self._maybe_regenerate_random_mask()
                    print(f"切换到面具: {self.current_mask_index + 1}")
                
                self.face_lost_frames = 0
                self.lost_notice_printed = False
            
            # 在每个人脸上应用面具
            if not used_landmarks:
                for (x, y, w, h) in faces:
                    frame = self.apply_mask_rect(frame, (x, y, w, h))
                
                # 绘制人脸矩形框（可选）
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            # 没有检测到人脸
            if self.face_detected:
                self.face_lost_frames += 1
                
                # 如果连续多帧检测不到人脸，认为脸被挡住了
                if self.face_lost_frames >= self.face_lost_threshold and not self.lost_notice_printed:
                    print("检测到脸部被遮挡，准备切换面具...")
                    self.lost_notice_printed = True
        
        # 在画面上显示信息（OpenCV 默认字体不支持中文，使用 PIL 以避免中文乱码）
        if self.text_renderer is not None and self.text_renderer.available:
            info_text = f"当前面具: {self.current_mask_index + 1}/{len(self.masks)}"
            mode_text = "(关键点拟合)" if MP_AVAILABLE else "(矩形拟合)"
            frame = self.text_renderer.draw_text(frame, f"{info_text} {mode_text}", (10, 30), font_size=26, color=(0, 255, 0))
            frame = self.text_renderer.draw_text(frame, "按 A/D 切换面具（第1个为随机），或用手遮挡脸再拿开", (10, 70), font_size=20, color=(0, 255, 0))
            frame = self.text_renderer.draw_text(frame, "按 'S' 截图，按 'Q' 或 ESC 退出", (10, 110), font_size=20, color=(0, 0, 255))
        else:
            # 回退到英文，尽量避免中文乱码
            info_text = f"Mask: {self.current_mask_index + 1}/{len(self.masks)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "A/D: switch mask; S: snapshot; Q/ESC: quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def _open_camera(self):
        """尝试打开多个摄像头索引，返回第一个可用的 VideoCapture。"""
        for idx in (0, 1, 2):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def run(self):
        """运行主程序"""
        cap = self._open_camera()
        
        if cap is None or not cap.isOpened():
            print("无法打开摄像头。\n- 请检查: 1) 摄像头是否被占用 2) 系统隐私权限是否允许应用访问摄像头 3) 若为台式机请确认摄像头连接\n- 如仍失败，可尝试在代码中将 VideoCapture(0) 改为 VideoCapture(1/2) 以尝试其他设备索引。")
            return
        
        print("程序开始运行...")
        print("- 确保在光线良好的环境中")
        print("- 正对摄像头")
        print("- 用手完全挡住脸部再拿开即可切换面具")
        print("- 按 'q' 键退出程序")
        
        snapshot_counter = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头画面")
                break
            
            # 处理帧
            processed_frame = self.process_frame(frame)
            
            # 显示结果
            cv2.imshow('川剧变脸 - Face Mask Changer', processed_frame)
            
            # 键盘交互：A/D 切换面具，S 截图，Q 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27: ESC
                break
            elif key == ord('a'):
                self.current_mask_index = (self.current_mask_index - 1) % len(self.masks)
                self._maybe_regenerate_random_mask()
                print(f"切换到面具: {self.current_mask_index + 1}")
            elif key == ord('d'):
                self.current_mask_index = (self.current_mask_index + 1) % len(self.masks)
                self._maybe_regenerate_random_mask()
                print(f"切换到面具: {self.current_mask_index + 1}")
            elif key == ord('s'):
                filename = f"snapshot_{snapshot_counter}.png"
                snapshot_counter += 1
                # 保存当前显示帧
                cv2.imwrite(filename, processed_frame)
                print(f"已保存截图: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    face_mask_changer = FaceMaskChanger()
    face_mask_changer.run()
