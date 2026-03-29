import torch
import os
from PIL import Image
from torchvision import transforms
from model import ViolenceClassifier

def test_folder(folder_path, model_path='./violence_model.pth'):
    # 1. Налаштування
    classes = ['NonViolence', 'Violence', 'Guns', 'Knife']
    img_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GPU-specific optimizations for inference
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    # ОНОВЛЕНО: Використовуємо нормалізацію ImageNet, як у тренувальному скрипті
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Завантажуємо модель один раз
    # ОНОВЛЕНО: use_pretrained=False, щоб не качати зайвого з мережі перед завантаженням своїх ваг
    model = ViolenceClassifier(num_classes=4, use_pretrained=False)
    
    # map_location дозволяє запускати модель на CPU, навіть якщо вона була навчена на GPU
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✓ Ваги моделі успішно завантажено!")
    except Exception as e:
        print(f"Помилка завантаження ваг: {e}")
        return

    model.to(device)
    model.eval()

    # 3. Перевіряємо, чи існує папка
    if not os.path.exists(folder_path):
        print(f"Помилка: Папка {folder_path} не знайдена.")
        return

    print(f"Починаю обробку зображень у папці: {folder_path}\n" + "-"*40)

    # 4. Перебір файлів
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    with torch.no_grad():
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(folder_path, filename)
                
                try:
                    # Завантаження та підготовка фото
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    # Прогноз
                    logits = model(img_tensor)
                    probs = torch.sigmoid(logits)[0]

                    # Вивід результатів
                    print(f"Файл: {filename}")
                    results = []
                    for i, class_name in enumerate(classes):
                        percentage = probs[i].item() * 100
                        # Виділяємо високу ймовірність (більше 50%)
                        if percentage > 50.0:
                            results.append(f"** {class_name}: {percentage:.1f}% **")
                        else:
                            results.append(f"{class_name}: {percentage:.1f}%")
                    
                    print(" | ".join(results))
                    print("-" * 40)
                
                except Exception as e:
                    print(f"Помилка при обробці {filename}: {e}")

if __name__ == '__main__':
    # Вкажи шлях до своєї папки з тестовими картинками
    test_folder('./test_images')