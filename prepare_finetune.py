import torch
import os

print("=== CHUẨN BỊ NÃO BỘ FINE-TUNING ===")
old_ckpt = "experiments/runs/20260322-184233/best_model.pth"
new_ckpt = "experiments/pretrained_base.pth"

if not os.path.exists(old_ckpt):
    print(f"Không tìm thấy {old_ckpt}. Vui lòng kiểm tra lại!")
    exit(1)
    
print(f"Đang nạp bộ nhớ 50 Epochs cũ ({old_ckpt})...")
checkpoint = torch.load(old_ckpt, map_location="cpu", weights_only=False)
state_dict = checkpoint["model_state_dict"]

# Xóa bỏ các nơ-ron liên quan đến số lượng nhãn cũ (21 nhãn)
# Vì bộ mới chỉ có 5 nhãn, PyTorch sẽ báo lỗi nếu không xóa đi để nó học lại
keys_to_delete = []
for k in state_dict.keys():
    # text_encoder dùng cho label
    if "text_encoder" in k:
        keys_to_delete.append(k)
    # classifier dùng cho xuất kết quả 21 class
    elif "classifier.3" in k: # Lớp tuyến tính cuối cùng
        keys_to_delete.append(k)
        
for k in keys_to_delete:
    del state_dict[k]
    
print(f"Đã cắt tỉa {len(keys_to_delete)} kết nối cũ không tương thích (chỉ giữ lại hệ thống trích xuất).")

# Chỉ lưu lại phần state_dict (đúng chuẩn Transfer Learning)
torch.save({"model_state_dict": state_dict}, new_ckpt)
print(f"🎉 Hoàn tất! Não bộ gốc đã được lưu tại: {new_ckpt}")
print("Bây giờ bạn có thể bắt đầu quá trình nạp kiến thức mới!")
