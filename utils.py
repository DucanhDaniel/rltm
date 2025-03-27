import os
import torch

def save_model(model, model_name = "model_weights", save_dir = "saved_model"):
    # Đường dẫn thư mục và file
    save_dir = "saved_model"
    save_path = os.path.join(save_dir, model_name + ".pth")

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(save_dir, exist_ok=True)

    # Lưu model
    torch.save(model.state_dict(), save_path)

    print(f"Model weights saved at {save_path}")


def load_model(model, model_name = "model_weights", save_dir = "saved_model", device = "cuda"):
    model.load_state_dict(torch.load(save_dir + "/model_weights.pth", weights_only=True))

    # Nếu dùng GPU mà muốn chạy trên CPU
    model.to("cuda")  # Nếu cần chuyển về CPU

    return model
