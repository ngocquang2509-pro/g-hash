import torch
from PIL import Image
from torchvision import transforms
from src.models.ghash import GHashModel
from src.utils.config import Config

def check():
    device = "cuda"
    config = Config("configs/et_edu_config.yaml")
    
    model = GHashModel(config.config)
    ckpt = torch.load("experiments/runs/20260329-193756/best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    imgs = [
        "images/hocsinh_crop_1.jpg",
        "images/hocsinh_crop_2.jpg",
        "images/dung_crop_0.jpg",
        "data/ET-EDU-CROPPED-PERSONS/D02_20240224090908_0038_person_00.jpg" # Rank 1 retrieved
    ]
    
    for p in imgs:
        img = Image.open(p).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.image_encoder(tensor)
            hash_c = model.generate_hash_code(tensor)
            logits = model.classifier(features)
            
        print(f"--- File: {p} ---")
        print(f"Logits 5 Class: {logits[0].tolist()}")
        print(f"Hash Output (sum): {hash_c.sum().item()} | Đầu băm: {hash_c[0][:10].tolist()}")
        
if __name__ == '__main__':
    check()
