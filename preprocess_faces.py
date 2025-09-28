import os
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
from torchvision.transforms import ToPILImage

def preprocess(src_dir="Data", dst_dir="processed", device="cpu"):
    mtcnn = MTCNN(keep_all=False, device=device)
    os.makedirs(dst_dir, exist_ok=True)
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    for cls in classes:
        in_dir = os.path.join(src_dir, cls)
        out_dir = os.path.join(dst_dir, cls)
        os.makedirs(out_dir, exist_ok=True)

        for fname in tqdm(os.listdir(in_dir), desc=f"Processing {cls}"):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, fname)

            try:
                img = Image.open(in_path).convert("RGB")
                face = mtcnn(img)

                if face is None:
                    img.resize((224, 224)).save(out_path)
                else:
                    pil = ToPILImage()(face.cpu())
                    pil.resize((224, 224)).save(out_path)

            except Exception as e:
                print("Error:", in_path, e)

if __name__ == "__main__":
    preprocess(src_dir="Data", dst_dir="processed", device="cpu")
