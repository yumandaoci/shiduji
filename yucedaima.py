'''
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import timm
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing

# ===================== 设置超参数 =====================
MODEL_PATH = 'best_model_convit.pth'  # 模型权重路径
IMAGE_FOLDER = 'D:\\redianou\\wendang\\tupianout'  # 图片文件夹路径
OUTPUT_DIR = 'D:\\redianou\\shuju'  # 输出文件夹
BATCH_SIZE = 16  # 批处理大小，可根据显存调整
DECIMAL_PLACES = 5  # 预测温度保留的小数位数
USE_GPU = torch.cuda.is_available()
# 修改：Windows平台建议设置为0，或者较小的数值
NUM_WORKERS = 0 if os.name == 'nt' else 4  # Windows使用0，Linux/Mac使用4

# 模型配置 - 根据实际训练的模型修改
MODEL_CONFIGS = {
    'maxvit': 'maxvit_tiny_tf_224',
    'convit': 'convit_base',
    'coatnet': 'coatnet_0_rw_224',
    'efficientvit': 'efficientvit_m4',
    'swin': 'swin_small_patch4_window7_224'
}

# 自动检测模型类型（根据保存的文件名）
def detect_model_type(model_path):
    """根据模型文件名自动检测模型类型"""
    model_name = os.path.basename(model_path).lower()
    for key in MODEL_CONFIGS.keys():
        if key in model_name:
            return MODEL_CONFIGS[key]
    return MODEL_CONFIGS['convit']  # 默认使用convit

# ===================== 数据集类 =====================
class ImageDataset(Dataset):
    """用于批量加载图片的数据集类"""

    def __init__(self, image_folder, transform=None, start_idx=0, end_idx=59313):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = []

        # 生成图片文件名列表 (0.jpg 到 59313.jpg)
        print(f"📋 准备图片列表 ({start_idx} 到 {end_idx})...")
        for i in range(start_idx, end_idx + 1):
            # 支持多种文件格式
            for ext in ['.jpg', '.jpeg', '.png']:
                image_path = os.path.join(image_folder, f"{i}{ext}")
                if os.path.exists(image_path):
                    self.image_files.append((i, image_path))
                    break
            else:
                # 如果文件不存在，记录但不中断
                print(f"⚠️ 图片 {i} 未找到")

        print(f"✅ 找到 {len(self.image_files)} 张有效图片")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_idx, image_path = self.image_files[idx]

        try:
            # 加载图片
            image = Image.open(image_path).convert("RGB")

            # 应用变换
            if self.transform:
                image = self.transform(image)

            return image, image_idx, os.path.basename(image_path)

        except Exception as e:
            print(f"❌ 加载图片 {image_path} 失败: {e}")
            # 返回黑色图片作为占位符
            dummy_image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, image_idx, f"error_{image_idx}.jpg"

def main():
    """主函数 - 所有主要逻辑都放在这里"""

    MODEL_NAME = detect_model_type(MODEL_PATH)
    device = torch.device("cuda" if USE_GPU else "cpu")
    print(f"使用设备: {device}")
    print(f"检测到模型类型: {MODEL_NAME}")
    print(f"批处理大小: {BATCH_SIZE}")
    print(f"数据加载线程数: {NUM_WORKERS}")
    print(f"预测温度保留小数位数: {DECIMAL_PLACES}")

    # ===================== 数据预处理 =====================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ===================== 加载模型 =====================
    print("⏱️ 开始加载模型...")
    model_load_start = time.time()

    # 创建模型
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1)
    model = model.to(device)

    # 加载权重
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 如果是完整的检查点
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 从检查点加载模型权重成功！")
                if 'model_name' in checkpoint:
                    print(f"📝 原始模型: {checkpoint['model_name']}")
            else:
                # 如果只是权重字典
                model.load_state_dict(checkpoint)
                print("✅ 模型权重加载成功！")
        except Exception as e:
            print(f"❌ 加载模型权重失败: {e}")
            print("🔄 尝试使用预训练权重...")
            model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1)
            model = model.to(device)
    else:
        raise FileNotFoundError(f"模型权重文件未找到: {MODEL_PATH}")

    model.eval()
    model_load_time = time.time() - model_load_start

    # GPU预热
    if USE_GPU:
        print("🔥 GPU预热中...")
        warmup_start = time.time()
        dummy_input = torch.randn(BATCH_SIZE, 3, 224, 224).to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        torch.cuda.empty_cache()
        warmup_time = time.time() - warmup_start
        print(f"📊 GPU预热时间: {warmup_time:.4f} 秒")

    # ===================== 创建数据集和数据加载器 =====================
    dataset = ImageDataset(IMAGE_FOLDER, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 保持顺序
        num_workers=NUM_WORKERS,
        pin_memory=USE_GPU,
        drop_last=False
    )

    print(f"📦 数据加载器创建完成，总批次数: {len(dataloader)}")

    # ===================== 批量预测 =====================
    print("\n🚀 开始批量预测...")
    total_start_time = time.time()

    results = []
    inference_times = []

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 使用tqdm显示进度
    with torch.no_grad():
        for batch_idx, (images, indices, filenames) in enumerate(tqdm(dataloader, desc="预测进度")):
            # 移动数据到设备
            images = images.to(device, non_blocking=True)

            # GPU同步（用于准确计时）
            if USE_GPU:
                torch.cuda.synchronize()

            # 推理计时
            inference_start = time.time()
            outputs = model(images)

            if USE_GPU:
                torch.cuda.synchronize()

            inference_time = time.time() - inference_start
            inference_times.append(inference_time)

            # 处理结果
            temperatures = outputs.cpu().numpy().flatten()

            # 保存结果，调整温度的小数位数
            for idx, temp, filename in zip(indices, temperatures, filenames):
                results.append({
                    'Index': idx.item(),
                    'Filename': filename,
                    'Predicted_Temperature': round(temp, DECIMAL_PLACES)
                })

            # 内存管理：每100个批次清理一次
            if (batch_idx + 1) % 100 == 0:
                if USE_GPU:
                    torch.cuda.empty_cache()
                gc.collect()

                # 中间保存（防止意外中断）
                if (batch_idx + 1) % 500 == 0:
                    temp_df = pd.DataFrame(results)
                    temp_csv_path = os.path.join(OUTPUT_DIR, f'temp_predictions_batch_{batch_idx + 1}.csv')
                    temp_df.to_csv(temp_csv_path, index=False)
                    print(f"💾 中间结果已保存: {temp_csv_path}")

    total_end_time = time.time()
    total_process_time = total_end_time - total_start_time

    print(f"\n🎉 批量预测完成！")
    print(f"📊 处理了 {len(results)} 张图片")
    print(f"⏱️ 总处理时间: {total_process_time:.2f} 秒")

    # ===================== 计算性能统计 =====================
    if inference_times:
        total_inference_time = sum(inference_times)
        avg_inference_time = total_inference_time / len(inference_times)
        total_images = len(results)

        print(f"\n📈 性能统计:")
        print(f"   总推理时间: {total_inference_time:.4f} 秒")
        print(f"   平均批次推理时间: {avg_inference_time:.4f} 秒")
        print(f"   平均单张图片推理时间: {total_inference_time / total_images * 1000:.2f} ms")
        print(f"   整体处理速度: {total_images / total_process_time:.2f} FPS")
        print(f"   纯推理速度: {total_images / total_inference_time:.2f} FPS")

    # ===================== 保存最终结果 =====================
    print("\n💾 保存预测结果...")

    # 按索引排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Index').reset_index(drop=True)

    # 保存预测结果
    predictions_csv_path = os.path.join(OUTPUT_DIR, 'batch_predictions_0_to_59313.csv')
    try:
        results_df.to_csv(predictions_csv_path, index=False)
        print(f"✅ 预测结果已保存到: {predictions_csv_path}")
    except Exception as e:
        print(f"❌ 保存预测结果失败: {e}")

    # 保存仅温度值的CSV（用于对比分析）
    temp_only_df = results_df[['Index', 'Predicted_Temperature']].copy()
    temp_only_df.columns = ['Index', 'Temperature']
    temp_only_csv_path = os.path.join(OUTPUT_DIR, 'predicted_temperatures_only.csv')

    try:
        temp_only_df.to_csv(temp_only_csv_path, index=False)
        print(f"✅ 温度预测值已保存到: {temp_only_csv_path}")
    except Exception as e:
        print(f"❌ 保存温度预测值失败: {e}")

    # 保存性能统计
    if inference_times:
        performance_stats = {
            'Total_Images': len(results),
            'Total_Process_Time_s': total_process_time,
            'Total_Inference_Time_s': total_inference_time,
            'Model_Load_Time_s': model_load_time,
            'Average_Batch_Inference_Time_s': avg_inference_time,
            'Average_Per_Image_Inference_ms': total_inference_time / len(results) * 1000,
            'Overall_FPS': len(results) / total_process_time,
            'Pure_Inference_FPS': len(results) / total_inference_time,
            'Batch_Size': BATCH_SIZE,
            'Device': str(device),
            'Model_Name': MODEL_NAME,
            'GPU_Available': USE_GPU,
            'Num_Workers': NUM_WORKERS,
            'Decimal_Places': DECIMAL_PLACES
        }

        performance_df = pd.DataFrame([performance_stats])
        performance_csv_path = os.path.join(OUTPUT_DIR, 'batch_inference_performance.csv')

        try:
            performance_df.to_csv(performance_csv_path, index=False)
            print(f"✅ 性能统计已保存到: {performance_csv_path}")
        except Exception as e:
            print(f"❌ 保存性能统计失败: {e}")

    # ===================== 数据验证和摘要 =====================
    print(f"\n📋 数据摘要:")
    print(f"   预测图片数量: {len(results)}")
    print(f"   索引范围: {results_df['Index'].min()} - {results_df['Index'].max()}")
    print(
        f"   温度预测范围: {results_df['Predicted_Temperature'].min():.1f}°C - {results_df['Predicted_Temperature'].max():.1f}°C")
    print(f"   平均预测温度: {results_df['Predicted_Temperature'].mean():.1f}°C")
    print(f"   温度标准差: {results_df['Predicted_Temperature'].std():.1f}°C")

    # 检查缺失的图片
    expected_indices = set(range(0, 59314))
    actual_indices = set(results_df['Index'])
    missing_indices = expected_indices - actual_indices

    if missing_indices:
        print(f"\n⚠️ 缺失的图片索引: {len(missing_indices)} 个")
        missing_df = pd.DataFrame({'Missing_Index': sorted(list(missing_indices))})
        missing_csv_path = os.path.join(OUTPUT_DIR, 'missing_images.csv')
        try:
            missing_df.to_csv(missing_csv_path, index=False)
            print(f"📝 缺失图片列表已保存到: {missing_csv_path}")
        except Exception as e:
            print(f"❌ 保存缺失图片列表失败: {e}")
    else:
        print("✅ 所有预期的图片都已处理完成")

    # 清理临时文件
    temp_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('temp_predictions_batch_')]
    for temp_file in temp_files:
        try:
            os.remove(os.path.join(OUTPUT_DIR, temp_file))
        except:
            pass

    print("\n🎉 程序执行完成！")

# ===================== 主程序入口 =====================
if __name__ == '__main__':
    # Windows multiprocessing支持
    multiprocessing.freeze_support()

    # 运行主函数
    main()

 '''
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import multiprocessing

# ===================== 设置超参数 =====================
MODEL_PATH = 'best_model.pth'  # 模型权重路径
IMAGE_FOLDER = 'D:\\shiduji\\tupian'  # 图片文件夹路径
OUTPUT_DIR = 'D:\\shiduji\\shuju'  # 输出文件夹
BATCH_SIZE = 16  # 批处理大小
DECIMAL_PLACES = 2  # 预测温度保留的小数位数
USE_GPU = torch.cuda.is_available()
NUM_WORKERS = 0 if os.name == 'nt' else 4  # Windows 使用 0，Linux/Mac 使用 4

# ===================== 数据集类 =====================
class ImageDataset(Dataset):
    """用于批量加载单通道红色图片的数据集类"""

    def __init__(self, image_folder, transform=None, start_idx=0, end_idx=14999):
        self.image_folder = image_folder
        self.transform = transform
        self.image_files = []

        # 生成图片文件名列表 (0.jpg 到 59313.jpg)
        print(f"📋 准备图片列表 ({start_idx} 到 {end_idx})...")
        for i in range(start_idx, end_idx + 1):
            for ext in ['.jpg', '.jpeg', '.png']:
                image_path = os.path.join(image_folder, f"{i}{ext}")
                if os.path.exists(image_path):
                    self.image_files.append((i, image_path))
                    break
            else:
                print(f"⚠️ 图片 {i} 未找到")

        print(f"✅ 找到 {len(self.image_files)} 张有效图片")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_idx, image_path = self.image_files[idx]

        try:
            # 加载单通道图片
            image = Image.open(image_path)
            if image.mode != 'L':
                print(f"警告: 图像 {image_path} 不是单通道 (mode={image.mode})，转换为单通道")
                image = image.convert('L')

            # 应用变换
            if self.transform:
                image = self.transform(image)

            return image, image_idx, os.path.basename(image_path)

        except Exception as e:
            print(f"❌ 加载图片 {image_path} 失败: {e}")
            # 返回黑色单通道图片作为占位符
            dummy_image = Image.new('L', (224, 224), 0)
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, image_idx, f"error_{image_idx}.jpg"

def main():
    """主函数 - 所有主要逻辑"""

    device = torch.device("cuda" if USE_GPU else "cpu")
    print(f"预测温度保留小数位数: {DECIMAL_PLACES}")

    # ===================== 数据预处理 =====================
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # ===================== 加载模型 =====================
    #print("⏱️ 开始加载模型...")
    model_load_start = time.time()

    # 创建 ResNet18 模型
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)

    # 加载权重
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ 从检查点加载模型权重成功！")
            else:
                model.load_state_dict(checkpoint)
                print("✅ 模型权重加载成功！")
        except Exception as e:
            print(f"❌ 加载模型权重失败: {e}")
            #print("🔄 使用随机初始化的 ResNet18...")
    else:
        print(f"⚠️ 模型权重文件未找到: {MODEL_PATH}")
        #print("🔄 使用随机初始化的 ResNet18...")

    model.eval()
    model_load_time = time.time() - model_load_start

    # GPU 预热
    if USE_GPU:
        #print("🔥 GPU 预热中...")
        warmup_start = time.time()
        dummy_input = torch.randn(BATCH_SIZE, 1, 224, 224).to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
        torch.cuda.empty_cache()
        warmup_time = time.time() - warmup_start
        #print(f"📊 GPU 预热时间: {warmup_time:.4f} 秒")

    # ===================== 创建数据集和数据加载器 =====================
    dataset = ImageDataset(IMAGE_FOLDER, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=USE_GPU,
        drop_last=False
    )

    #print(f"📦 数据加载器创建完成，总批次数: {len(dataloader)}")

    # ===================== 批量预测 =====================
    #print("\n🚀 开始批量预测...")
    total_start_time = time.time()

    results = []
    inference_times = []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, indices, filenames) in enumerate(tqdm(dataloader, desc="预测进度")):
            images = images.to(device, non_blocking=True)

            if USE_GPU:
                torch.cuda.synchronize()

            inference_start = time.time()
            outputs = model(images)

            if USE_GPU:
                torch.cuda.synchronize()

            inference_time = time.time() - inference_start
            inference_times.append(inference_time)

            temperatures = outputs.cpu().numpy().flatten()

            for idx, temp, filename in zip(indices, temperatures, filenames):
                results.append({
                    'Index': idx.item(),
                    'Filename': filename,
                    'Predicted_Temperature': round(temp, DECIMAL_PLACES)
                })

            if (batch_idx + 1) % 100 == 0:
                if USE_GPU:
                    torch.cuda.empty_cache()
                gc.collect()

                if (batch_idx + 1) % 500 == 0:
                    temp_df = pd.DataFrame(results)
                    temp_csv_path = os.path.join(OUTPUT_DIR, f'temp_predictions_batch_{batch_idx + 1}.csv')
                    temp_df.to_csv(temp_csv_path, index=False)
                    print(f"💾 中间结果已保存: {temp_csv_path}")

    total_end_time = time.time()
    total_process_time = total_end_time - total_start_time

    #print(f"\n🎉 批量预测完成！")
    #print(f"📊 处理了 {len(results)} 张图片")
    #print(f"⏱️ 总处理时间: {total_process_time:.2f} 秒")

    # ===================== 计算性能统计 =====================
    if inference_times:
        total_inference_time = sum(inference_times)
        avg_inference_time = total_inference_time / len(inference_times)
        total_images = len(results)
        print(f"   总推理时间: {total_inference_time:.4f} 秒")

    # ===================== 保存最终结果 =====================
    print("\n💾 保存预测结果...")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Index').reset_index(drop=True)

    predictions_csv_path = os.path.join(OUTPUT_DIR, 'batch_predictions_0_to_14999.csv')
    try:
        results_df.to_csv(predictions_csv_path, index=False)
        print(f"✅ 预测结果已保存到: {predictions_csv_path}")
    except Exception as e:
        print(f"❌ 保存预测结果失败: {e}")

    temp_only_df = results_df[['Index', 'Predicted_Temperature']].copy()
    temp_only_df.columns = ['Index', 'Temperature']
    temp_only_csv_path = os.path.join(OUTPUT_DIR, 'predicted_temperatures_only.csv')

    try:
        temp_only_df.to_csv(temp_only_csv_path, index=False)
        print(f"✅ 温度预测值已保存到: {temp_only_csv_path}")
    except Exception as e:
        print(f"❌ 保存温度预测值失败: {e}")

    if inference_times:
        performance_stats = {
            'Total_Images': len(results),
            'Num_Workers': NUM_WORKERS,
            'Decimal_Places': DECIMAL_PLACES
        }

        performance_df = pd.DataFrame([performance_stats])
        performance_csv_path = os.path.join(OUTPUT_DIR, 'batch_inference_performance.csv')

        try:
            performance_df.to_csv(performance_csv_path, index=False)
            print(f"✅ 性能统计已保存到: {performance_csv_path}")
        except Exception as e:
            print(f"❌ 保存性能统计失败: {e}")


    expected_indices = set(range(0,14999))
    actual_indices = set(results_df['Index'])
    missing_indices = expected_indices - actual_indices

    if missing_indices:
        print(f"\n⚠️ 缺失的图片索引: {len(missing_indices)} 个")
        missing_df = pd.DataFrame({'Missing_Index': sorted(list(missing_indices))})
        missing_csv_path = os.path.join(OUTPUT_DIR, 'missing_images.csv')
        try:
            missing_df.to_csv(missing_csv_path, index=False)
            print(f"📝 缺失图片列表已保存到: {missing_csv_path}")
        except Exception as e:
            print(f"❌ 保存缺失图片列表失败: {e}")
    else:
        print("✅ 所有预期的图片都已处理完成")

    temp_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('temp_predictions_batch_')]
    for temp_file in temp_files:
        try:
            os.remove(os.path.join(OUTPUT_DIR, temp_file))
        except:
            pass

    print("\n🎉 程序执行完成！")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()