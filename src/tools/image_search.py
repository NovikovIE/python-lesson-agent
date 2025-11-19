import os
import requests
from io import BytesIO
from typing import List, Optional
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS

class ImageSearchTool:
    def __init__(self, save_folder: str = "downloaded_images"):
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        
        print("[image search] Loading CLIP Models")
        self.img_model = SentenceTransformer('clip-ViT-B-32')
        
        self.text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
        print("[image search] Models Loaded ---")

    def search_and_rerank(self, query: str, search_limit: int = 5) -> Optional[str]:
        print(f"[image search] Searching images for: '{query}'...")
        
        image_results = []
        try:
            with DDGS() as ddgs:
                results = ddgs.images(query, max_results=search_limit + 3)
                image_results = [r['image'] for r in results]
        except Exception as e:
            print(f"[image search] Error searching DuckDuckGo: {e}")
            return None

        if not image_results:
            return None

        candidates_pil = []
        candidates_urls = []

        for url in image_results:
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content)).convert("RGB")
                    candidates_pil.append(img)
                    candidates_urls.append(url)
            except Exception:
                continue
            
            if len(candidates_pil) >= search_limit:
                break
        
        if not candidates_pil:
            print("[image search] !!!WARNING!!! No images could be downloaded.")
            return None

        print(f"[image search] Downloaded {len(candidates_pil)} candidates. Reranking...")

        text_emb = self.text_model.encode(query, convert_to_tensor=True)
        
        img_embs = self.img_model.encode(candidates_pil, convert_to_tensor=True)

        scores = util.cos_sim(text_emb, img_embs)[0]

        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
        best_image = candidates_pil[best_idx]

        print(f"[image search] Best match score: {best_score:.4f}")

        filename = f"{query.replace(' ', '_')[:30]}_{int(best_score*100)}.jpg"
        filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in ('_', '.')])
        
        save_path = os.path.join(self.save_folder, filename)
        best_image.save(save_path)
        
        return save_path

image_tool = ImageSearchTool()
