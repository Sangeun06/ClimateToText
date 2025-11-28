import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import numpy as np
import json
import logging

from tqdm import tqdm
import pdb

from torchvision import transforms
from typing import Optional, List


# --- 1. 헬퍼(Helper) 함수들 ---
def pil_loader(path: Path) -> Image.Image:
    """일반적인 이미지 파일(png, jpg)을 PIL Image 객체로 로드합니다."""
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as e:
        logging.warning(f"PIL 로드 실패 {path}: {e}. 검은색 이미지 반환.")
        return Image.new("RGB", (224, 224), (0, 0, 0))

def tensor_loader(path: Path) -> torch.Tensor:
    """직렬화된 텐서(.npy, .pt)를 로드합니다."""
    ext = path.suffix.lower()
    try:
        if ext == ".npy":
            data = np.load(path)
        elif ext == ".pt" or ext == ".pth":
            data = torch.load(path, map_location="cpu")
        else:
            raise ValueError(f"지원하지 않는 텐서 확장자: {ext}")

        return torch.from_numpy(data).float() if isinstance(data, np.ndarray) else data.float()
    except Exception as e:
        logging.warning(f"텐서 로드 실패 {path}: {e}. 빈 텐서 반환.")
        return torch.zeros(3, 224, 224) # (채널, H, W) - 임시 크기

def get_default_image_transform(size: int = 224) -> transforms.Compose:
    '''
    """S2, Mesoscale 차트 등 일반 이미지용 기본 트랜스폼 (ImageNet 정규화)"""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    '''
    # MAE는 일반적으로 ImageNet 정규화를 '사용하지 않음'
    # 픽셀 값을 [0, 1]로 스케일링한 뒤 -0.5 ~ 0.5 범위로 정규화
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def get_default_tensor_transform(size: int = 224) -> transforms.Compose:
    """ERA5 텐서용 기본 트랜스폼 (리사이즈만 수행)"""
    # 참고: ERA5 데이터는 채널별로 별도의 정규화(Normalize)가 필요할 수 있습니다.
    # 우선 리사이즈만 적용합니다.
    return transforms.Compose([
        transforms.Resize((size, size)),
    ])


# --- 2. 추상 기본 클래스: MultiTaskImageDataset (Stage 1)
class MultiTaskImageDataset(Dataset):
    """
    Stage 1용 멀티 태스크 이미지 데이터셋.
    - WeatherQA의 para_paths를 낱개 이미지 샘플로 펼치고, 타입명을 조건(cond_name)으로 사용
    - 일반 ImageFolder(root/class/*.jpg)도 조건(cond_name)으로 사용
    - cond_name을 전역 type_map으로 통합하여 다양한 데이터셋을 한 번에 학습 가능

    반환 샘플:
      { 'image': Tensor(C,H,W), 'cond_id': int, 'cond_name': str }
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        weatherqa_json_path: Optional[str] = None,
        weatherqa_image_root: Optional[str] = None,
        weatherqa_years: Optional[List[int]] = None,
        chatearthnet_json_path: Optional[str] = None,
        chatearthnet_image_root: Optional[str] = None,
        climateiqa_json_path: Optional[str] = None,
        climateiqa_image_root: Optional[str] = None,
        extra_json_path: Optional[str] = None,
        extra_image_root: Optional[str] = None,
        image_size: int = 224,
        weatherqa_include_types: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        types = []

        self.transform = get_default_image_transform(size=image_size)
        self.tensor_transform = get_default_tensor_transform(size=image_size)
        # items: heterogeneous records
        # ('image', path, cond_name) OR ('era5_channel', path, channel_index, cond_name)
        self.items: list[tuple] = []
        self.weatherqa_include_types = weatherqa_include_types

        # 1. WeatherQA 소스 수집
        weatherqa_record_cnt = 0
        chatearthnet_record_cnt = 0
        climateiqa_record_cnt = 0

        if weatherqa_json_path and weatherqa_image_root:
            try:
                with open(weatherqa_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                image_root = Path(weatherqa_image_root)
                def extract_year(p: str) -> Optional[int]:
                    for part in Path(p).parts:
                        if part.isdigit() and len(part) == 4:
                            try:
                                y = int(part)
                                if 1900 <= y <= 2100:
                                    return y
                            except Exception:
                                pass
                    return None

                for rec in (data.values() if isinstance(data, dict) else data):
                    for p in rec.get("para_paths", []):
                        if weatherqa_years is not None:
                            y = extract_year(p)
                            if (y is None) or (y not in weatherqa_years):
                                continue
                        full = image_root / p
                        tname = Path(p).parent.name
                        types.append(f"wqa:{tname}")
                        # 특정 종류 이미지만 처리
                        if self.weatherqa_include_types is not None:
                            if tname not in self.weatherqa_include_types:
                                continue
                        if full.exists() and tname:
                            self.items.append(("image", full, f"wqa:{tname}"))
                weatherqa_record_cnt = len(self.items)
                logging.info(f"WeatherQA 로드 완료: {weatherqa_record_cnt} 레코드")
            except Exception as e:
                logging.warning(f"MultiTaskImageDataset WeatherQA 로드 실패: {e}")

        # 2. ChatEarthNet 소스 수집 (3 밴드 이미지 세트 → 개별 이미지로)
        if chatearthnet_json_path and chatearthnet_image_root:
            try:
                with open(chatearthnet_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                image_root = Path(chatearthnet_image_root)
                records = data.values() if isinstance(data, dict) else data
                for rec in records:
                    base_path_str = rec.get("image")
                    if not base_path_str:
                        continue
                    filename = Path(base_path_str).name
                    path_rgb = image_root / "s2_rgb_images" / filename
                    path_b567 = image_root / "s2_band_5_6_7_images" / filename
                    path_b8_11_12 = image_root / "s2_band_8_11_12_images" / filename
                    if path_rgb.exists():
                        self.items.append(("image", path_rgb, "s2:RGB"))
                    if path_b567.exists():
                        self.items.append(("image", path_b567, "s2:B567"))
                    if path_b8_11_12.exists():
                        self.items.append(("image", path_b8_11_12, "s2:B8_11_12"))
                chatearthnet_record_cnt = len(self.items) - weatherqa_record_cnt
                logging.info(f"ChatEarthNet 로드 완료: {chatearthnet_record_cnt} 레코드")
            except Exception as e:
                logging.warning(f"MultiTaskImageDataset ChatEarthNet 로드 실패: {e}")

        # 3. ClimateIQA 소스 수집 (3채널 텐서 → 채널별 샘플)
        if climateiqa_json_path and climateiqa_image_root:
            try:
                with open(climateiqa_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                image_root = Path(climateiqa_image_root)
                records = data.values() if isinstance(data, dict) else data
                for rec in records:
                    # ClimateIQA caption 데이터 추출
                    query = rec.get("query")
                    if query is None:
                        continue
                    is_annotation = True if "describe" in query.lower() else False
                    if not is_annotation:
                        continue
                    tensor_rel = rec.get("images")[0]
                    if not tensor_rel:
                        continue
                    full = image_root / tensor_rel
                    heatmap_type = tensor_rel.split("/")[-1].split("_heatmap")[0]
                    if full.exists():
                        # channel indices: 0: Gust, 1: Precip, 2: Temp
                        channel_idx = 0 if heatmap_type == "wind_gust" else 1 if heatmap_type == "image_tp" else 2
                        type_key = "era5:Gust" if channel_idx == 0 else "era5:Precip" if channel_idx == 1 else "era5:Temp"
                        #self.items.append(("era5_channel", full, channel_idx, type_key))
                        self.items.append(("image", full, type_key))
                climateiqa_record_cnt = len(self.items) - (chatearthnet_record_cnt + weatherqa_record_cnt)
                logging.info(f"ClimateIQA 로드 완료: {climateiqa_record_cnt} 레코드")
            except Exception as e:
                logging.warning(f"MultiTaskImageDataset ClimateIQA 로드 실패: {e}")

        # 4. 추가 이미지-텍스트 페어 소스 수집
        if extra_json_path and extra_image_root:
            try:
                with open(extra_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                image_root = Path(extra_image_root)
                for rec in (data.values() if isinstance(data, dict) else data):
                    img_path = rec.get("image") # 이미지는 절대 경로
                    type_key = rec.get("cond_name", "extra:unknown")
                    if not img_path:
                        continue
                    full = Path(img_path)
                    types.append(type_key)
                    if full.exists():
                        self.items.append(("image", full, type_key))
                logging.info(f"추가 데이터 로드 완료: {len(self.items) - (climateiqa_record_cnt + chatearthnet_record_cnt + weatherqa_record_cnt)} 레코드")
            except Exception as e:
                logging.warning(f"MultiTaskImageDataset 추가 데이터 로드 실패: {e}")

        if not self.items:
            logging.warning("MultiTaskImageDataset: 수집된 이미지가 없습니다.")

        # 전역 type_map 구축
        self.type_map: dict[str, int] = {n: i for i, n in enumerate(sorted(set(types)))}
        # 더미 타입 추가
        for i in range(20 - len(self.type_map)):
            __spec__ = f"dummy_type_{i}"
            if __spec__ not in self.type_map:
                self.type_map[__spec__] = i + len(self.type_map)
        logging.info(f"총 {len(self.type_map)} 조건 타입이 구축되었습니다.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        rec = self.items[idx]
        kind = rec[0]
        if kind == "image":
            _, path, cond_name = rec
            try:
                img = pil_loader(path)
                img = self.transform(img)
            except Exception as e:
                logging.warning(f"이미지 로드 실패 {path}: {e}. 0 텐서 대체.")
                img = torch.zeros(3, 224, 224, dtype=torch.float32)
        elif kind == "era5_channel":
            _, path, ch_idx, cond_name = rec
            try:
                tensor = tensor_loader(path)  # (C,H,W) expected
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(0)
                if tensor.size(0) <= ch_idx:
                    logging.warning(f"ERA5 채널 부족: {path} 채널 {tensor.size(0)} < 요청 {ch_idx}")
                    # fallback zero
                    tensor_ch = torch.zeros(1, 224, 224)
                else:
                    tensor_ch = tensor[ch_idx:ch_idx+1, ...]  # (1,H,W)
                # resize
                # get_default_tensor_transform uses torchvision Resize which expects PIL or Tensor CHW with float
                tensor_ch = self.tensor_transform(tensor_ch)
                # (1,H,W) -> (3,H,W)
                img = tensor_ch.repeat(3, 1, 1)
            except Exception as e:
                logging.warning(f"ERA5 로드 실패 {path}: {e}. 0 텐서 대체.")
                img = torch.zeros(3, 224, 224, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown record kind: {kind}")

        cond_id = self.type_map.get(cond_name, 0)
        return {"image": img, "cond_id": cond_id, "cond_name": cond_name}

    def load_data_items(self, load_path: str) -> None:
        """JSON 파일에서 데이터 항목을 로드합니다."""
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data_loaded = json.load(f)

            self.items = []
            for entry in data_loaded:
                # 특정 종류 이미지만 처리
                tname = entry["cond_name"].split(":")[-1]
                if self.weatherqa_include_types is not None:
                    if tname not in self.weatherqa_include_types:
                        continue
                image_path = Path(entry["image"])
                cond_name = entry["cond_name"]
                self.items.append(("image", image_path, cond_name))

            logging.info(f"{load_path}에서 데이터 항목이 로드되었습니다. 총 {len(self.items)} 항목.")

        except Exception as e:
            logging.error(f"데이터 항목 로드 실패 {load_path}: {e}")

class ImageTextPairDataset(Dataset):
    """
    이미지-텍스트 쌍 데이터셋.

    반환 샘플:
      { 'image': Tensor(C,H,W), 'annotation': str, 'cond_name': str }
    """
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        weatherqa_json_path: Optional[str] = None,
        weatherqa_image_root: Optional[str] = None,
        weatherqa_years: Optional[List[int]] = None,
        climateiqa_json_path: Optional[str] = None,
        climateiqa_image_root: Optional[str] = None,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        self.transform = get_default_image_transform(size=image_size)
        # items: (image, annotation, cond_name) records
        self.items: list[tuple] = []

        # 1. WeatherQA 소스 수집
        weatherqa_record_cnt = 0
        climateiqa_record_cnt = 0
        if weatherqa_json_path and weatherqa_image_root:
            try:
                with open(weatherqa_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                image_root = Path(weatherqa_image_root)
                def extract_year(p: str) -> Optional[int]:
                    for part in Path(p).parts:
                        if part.isdigit() and len(part) == 4:
                            try:
                                y = int(part)
                                if 1900 <= y <= 2100:
                                    return y
                            except Exception:
                                pass
                    return None

                for rec in (data.values() if isinstance(data, dict) else data):
                    for p in rec.get("para_paths", []):
                        if weatherqa_years is not None:
                            y = extract_year(p)
                            if (y is None) or (y not in weatherqa_years):
                                continue
                        full = image_root / p
                        tname = Path(p).parent.name
                        if full.exists() and tname:
                            text = rec.get("annotations", "")
                            self.items.append((full, text, f"wqa:{tname}"))
                weatherqa_record_cnt = len(self.items)
                logging.info(f"WeatherQA 로드 완료: {weatherqa_record_cnt} 레코드")
            except Exception as e:
                logging.warning(f"ImageTextPairDataset WeatherQA 로드 실패: {e}")

        # 2. ClimateIQA 소스 수집 (3채널 텐서 → 채널별 샘플)
        if climateiqa_json_path and climateiqa_image_root:
            try:
                with open(climateiqa_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                image_root = Path(climateiqa_image_root)
                records = data.values() if isinstance(data, dict) else data
                for rec in records:
                    # ClimateIQA caption 데이터 추출
                    query = rec.get("query")
                    if query is None:
                        continue
                    is_annotation = True if "describe" in query.lower() else False
                    if not is_annotation:
                        continue
                    tensor_rel = rec.get("images")[0]
                    if not tensor_rel:
                        continue
                    full = image_root / tensor_rel
                    heatmap_type = tensor_rel.split("/")[-1].split("_heatmap")[0]
                    if full.exists():
                        # channel indices: 0: Gust, 1: Precip, 2: Temp
                        channel_idx = 0 if heatmap_type == "wind_gust" else 1 if heatmap_type == "image_tp" else 2
                        type_key = "era5:Gust" if channel_idx == 0 else "era5:Precip" if channel_idx == 1 else "era5:Temp"
                        text = rec.get("response", "")
                        self.items.append((full, text, type_key))
                climateiqa_record_cnt = len(self.items) - (weatherqa_record_cnt)
                logging.info(f"ClimateIQA 로드 완료: {climateiqa_record_cnt} 레코드")
            except Exception as e:
                logging.warning(f"MultiTaskImageDataset ClimateIQA 로드 실패: {e}")

        if not self.items:
            logging.warning("ImageTextPairDataset: 수집된 이미지가 없습니다.")

        # 전역 type_map 구축
        cond_names = sorted({rec[-1] for rec in self.items})
        self.type_map: dict[str, int] = {n: i for i, n in enumerate(cond_names)}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        rec = self.items[idx]
        path, cond_name = rec
        try:
            img = pil_loader(path)
            img = self.transform(img)
        except Exception as e:
            logging.warning(f"이미지 로드 실패 {path}: {e}. 0 텐서 대체.")
            img = torch.zeros(3, 224, 224, dtype=torch.float32)

        cond_id = self.type_map.get(cond_name, 0)
        return {"image": img, "cond_id": cond_id, "cond_name": cond_name}

    def save_data_items(self, save_path: str) -> None:
        """수집된 데이터 항목을 JSON 파일로 저장합니다."""
        try:
            whole_data_to_save = []
            for item in tqdm(self.items):
                #img = pil_loader(str(item[0]))
                data_to_save = [
                    {
                        "image": str(item[0]),
                        "annotation": item[1],
                        "cond_name": item[2]
                    }
                ]
                whole_data_to_save.extend(data_to_save)

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(whole_data_to_save, f, indent=4)
            logging.info(f"데이터 항목이 {save_path}에 저장되었습니다.")
            logging.info(f"총 {len(self.items)} 항목 저장됨.")

        except Exception as e:
            logging.error(f"데이터 항목 저장 실패 {save_path}: {e}")

    def load_data_items(self, load_path: str) -> None:
        """JSON 파일에서 데이터 항목을 로드합니다."""
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data_loaded = json.load(f)

            self.items = []
            for entry in data_loaded:
                image_path = Path(entry["image"])
                annotation = entry["annotation"]
                cond_name = entry["cond_name"]
                self.items.append((image_path, annotation, cond_name))

            logging.info(f"{load_path}에서 데이터 항목이 로드되었습니다. 총 {len(self.items)} 항목.")

        except Exception as e:
            logging.error(f"데이터 항목 로드 실패 {load_path}: {e}")

def collate_images(batch: List[dict]) -> dict:
    """Collate for Stage1 image-only datasets.
    - Pads images in a batch to the same H,W if needed and stacks to (B,C,H,W).
    - Returns `images` tensor and `cond_ids` (from `cond_id` or `type_id`).
    """
    # filter out None entries
    batch = [b for b in batch if b is not None]
    if not batch:
        return {"images": torch.empty(0), "cond_ids": torch.empty(0, dtype=torch.long)}

    imgs = [b["image"] for b in batch]
    # If all shapes equal, stack directly
    same_size = all(img.shape == imgs[0].shape for img in imgs)
    if same_size:
        images = torch.stack(imgs)
    else:
        # pad to max H,W
        h_max = max(img.shape[-2] for img in imgs)
        w_max = max(img.shape[-1] for img in imgs)
        padded = []
        for img in imgs:
            _, h, w = img.shape
            pad_h = h_max - h
            pad_w = w_max - w
            pad = (0, pad_w, 0, pad_h)
            padded.append(F.pad(img, pad, value=0.0))
        images = torch.stack(padded)

    # cond ids
    if "cond_id" in batch[0]:
        cond_ids = torch.tensor([b["cond_id"] for b in batch], dtype=torch.long)
    else:
        cond_ids = torch.tensor([b.get("type_id", 0) for b in batch], dtype=torch.long)

    return {"images": images, "cond_ids": cond_ids}