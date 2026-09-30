"""
Weave Dataset Utilities for Fashion Multimodal Training
Supports:
1. Real Hugging Face Fashion Dataset (ashraq/fashion-product-images-small):
   ~44,000 real e-commerce fashion catalog photographs (filtered for real Apparel:
   Dresses, Jackets, Tops, Shirts, Pants, Skirts, Coats).
2. Local directory image-text datasets with metadata.jsonl / metadata.csv (Fashion-Gen / DeepFashion format)
3. Fallback procedural demo generator if offline
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
try:
    from transformers import CLIPTokenizer
except ImportError:
    CLIPTokenizer = Any  # type: ignore

# Standard Fashion Taxonomy for Weave
FASHION_CATEGORIES = [
    "Dress",
    "Jacket",
    "Coat",
    "Pants",
    "Shirt",
    "Top",
    "Skirt",
    "Sweater",
    "Suit",
    "Jumpsuit",
    "Shorts",
    "Hoodie"
]

CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(FASHION_CATEGORIES)}
IDX_TO_CATEGORY = {idx: cat for idx, cat in enumerate(FASHION_CATEGORIES)}

ARTICLE_TYPE_TO_CATEGORY = {
    "Dresses": "Dress",
    "Jackets": "Jacket",
    "Coats": "Coat",
    "Blazers": "Suit",
    "Suits": "Suit",
    "Shirts": "Shirt",
    "Tshirts": "Top",
    "Tops": "Top",
    "Tunics": "Top",
    "Skirts": "Skirt",
    "Sweaters": "Sweater",
    "Sweatshirts": "Hoodie",
    "Jeans": "Pants",
    "Trousers": "Pants",
    "Track Pants": "Pants",
    "Shorts": "Shorts",
    "Jumpsuit": "Jumpsuit",
    "Rompers": "Jumpsuit"
}

class FashionDiffusionDataset(Dataset):
    """
    Dataset for Stable Diffusion 1.5 LoRA training.
    Loads real fashion photographs from Hugging Face or local directories.
    """
    def __init__(
        self,
        data_dir: Optional[str] = None,
        tokenizer: Optional[CLIPTokenizer] = None,
        size: int = 512,
        center_crop: bool = True,
        use_real_hf_dataset: bool = True,
        max_samples: Optional[int] = 2000,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.items: List[Dict[str, Any]] = []

        # Image preprocessing for Diffusion (values in [-1, 1])
        crop_fn = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            crop_fn,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Priority 1: User's local dataset (e.g. uploaded Fashion-Gen in Google Drive)
        if data_dir and os.path.exists(data_dir):
            print(f"[Dataset] Loading from local directory: {data_dir}")
            self._load_from_directory(data_dir)

        # Priority 2: Real Hugging Face Fashion Dataset (ashraq/fashion-product-images-small)
        if len(self.items) == 0 and use_real_hf_dataset:
            try:
                print("[Dataset] Loading REAL fashion photographs from Hugging Face: ashraq/fashion-product-images-small...")
                from datasets import load_dataset
                hf_ds = load_dataset("ashraq/fashion-product-images-small", split="train")
                print(f"[Dataset] Downloaded {len(hf_ds)} real fashion products. Filtering for Apparel...")

                count = 0
                for row in hf_ds:
                    # Filter for real clothing apparel
                    if row.get("masterCategory") == "Apparel":
                        art = row.get("articleType", "")
                        cat = ARTICLE_TYPE_TO_CATEGORY.get(art, "Top")
                        col = row.get("baseColour", "")
                        name = row.get("productDisplayName", f"{col} {cat}")
                        
                        prompt = f"a high-end designer {col.lower()} {name.lower()}, studio fashion photography, neutral background"
                        self.items.append({
                            "pil_image": row["image"],
                            "text": prompt,
                            "category_idx": CATEGORY_TO_IDX[cat],
                            "category": cat
                        })
                        count += 1
                        if max_samples and count >= max_samples:
                            break

                print(f"[Dataset] Successfully loaded {len(self.items)} REAL fashion garments (Dresses, Jackets, Pants, Tops, etc.).")
            except Exception as e:
                print(f"[Dataset Warning] Could not load Hugging Face dataset: {e}")

        # Priority 3: Fallback procedural demo if offline
        if len(self.items) == 0:
            print("[Dataset] Fallback: Generating procedural fashion concept samples...")
            self._generate_demo_samples(num_samples=max_samples or 120)

    def _load_from_directory(self, data_dir: str):
        metadata_jsonl = os.path.join(data_dir, "metadata.jsonl")
        metadata_csv = os.path.join(data_dir, "metadata.csv")

        if os.path.exists(metadata_jsonl):
            with open(metadata_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        img_path = os.path.join(data_dir, entry.get("file_name", entry.get("image", "")))
                        if os.path.exists(img_path):
                            text = entry.get("text", entry.get("prompt", "a high fashion garment"))
                            category = entry.get("category", "Dress")
                            cat_idx = CATEGORY_TO_IDX.get(category, 0)
                            self.items.append({
                                "image_path": img_path,
                                "text": text,
                                "category_idx": cat_idx,
                                "category": category
                            })
        elif os.path.exists(metadata_csv):
            import pandas as pd
            df = pd.read_csv(metadata_csv)
            for _, row in df.iterrows():
                img_col = "file_name" if "file_name" in row else ("image" if "image" in row else df.columns[0])
                text_col = "text" if "text" in row else ("prompt" if "prompt" in row else df.columns[1])
                img_path = os.path.join(data_dir, str(row[img_col]))
                if os.path.exists(img_path):
                    text = str(row[text_col])
                    cat = row.get("category", "Dress") if "category" in row else "Dress"
                    cat_idx = CATEGORY_TO_IDX.get(cat, 0)
                    self.items.append({
                        "image_path": img_path,
                        "text": text,
                        "category_idx": cat_idx,
                        "category": cat
                    })

    def _generate_demo_samples(self, num_samples: int = 120):
        cache_dir = os.path.expanduser("~/.cache/weave_demo_data")
        os.makedirs(cache_dir, exist_ok=True)
        colors = [("terracotta", (196, 87, 59)), ("bone white", (235, 230, 220)), ("charcoal black", (33, 29, 24)), ("navy", (28, 38, 56))]
        for i in range(num_samples):
            cat = FASHION_CATEGORIES[i % len(FASHION_CATEGORIES)]
            col_name, col_rgb = random.choice(colors)
            p = os.path.join(cache_dir, f"sample_{i:04d}_{cat}.png")
            if not os.path.exists(p):
                img = Image.new("RGB", (self.size, self.size), color=(240, 236, 230))
                draw = ImageDraw.Draw(img)
                cx, cy = self.size // 2, self.size // 2
                draw.polygon([(cx - 70, cy - 90), (cx + 70, cy - 90), (cx + 120, cy + 180), (cx - 120, cy + 180)], fill=col_rgb)
                img.save(p)
            text = f"a high-end designer {col_name} {cat.lower()}, studio fashion photography"
            self.items.append({"image_path": p, "text": text, "category_idx": CATEGORY_TO_IDX[cat], "category": cat})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        if "pil_image" in item:
            image = item["pil_image"].convert("RGB")
        else:
            image = Image.open(item["image_path"]).convert("RGB")

        pixel_values = self.image_transforms(image)

        result: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "category_idx": torch.tensor(item["category_idx"], dtype=torch.long),
            "text": item["text"]
        }

        if self.tokenizer is not None:
            inputs = self.tokenizer(
                item["text"],
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            result["input_ids"] = inputs.input_ids[0]

        return result


class FashionClassifierDataset(Dataset):
    """Dataset for training Category Classifier on real fashion images."""
    def __init__(self, base_dataset: FashionDiffusionDataset, is_train: bool = True):
        self.items = base_dataset.items
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224) if not is_train else transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip() if is_train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.items[idx]
        if "pil_image" in item:
            img = item["pil_image"].convert("RGB")
        else:
            img = Image.open(item["image_path"]).convert("RGB")
        return self.transform(img), item["category_idx"]
