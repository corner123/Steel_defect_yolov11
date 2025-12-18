from ultralytics import YOLO
import warnings
DATA_PATH = r"NEU-DET/steel_defect.yaml"
# 忽略一些不影响训练的警告
warnings.filterwarnings('ignore')

def main():
    print("🚀 正在初始化魔改版 YOLOv11-ema...")

    # ---------------------------------------------------
    # 1. 构建模型骨架 (关键步骤！)
    # ---------------------------------------------------
    # 这里必须加载你的 yaml 文件，而不是 .pt 文件
    # 因为我们要告诉程序：按我看好的新图纸（加了新模块的）来造模型
    model = YOLO('yolov11n_ema.yaml')

    # ---------------------------------------------------
    # 2. 加载预训练权重 (迁移学习)
    # ---------------------------------------------------
    # 虽然结构改了，但前面几层卷积（Backbone）还是通用的。
    # 加载官方权重可以让模型不用从零学起，收敛快很多。
    # ⚠️ 注意：运行这行时控制台会报红色/黄色的警告，说 "Shape mismatch" 或 "Missing keys"
    # 这是完全正常的！因为你加了新层，它找不到对应的权重，就会跳过。不用管它。
    try:
        model.load('yolo11n.pt')
        print("✅ 成功加载预训练权重（部分不匹配层已自动跳过）")
    except Exception as e:
        print(f"⚠️ 权重加载提示: {e}")

    # ---------------------------------------------------
    # 3. 开始训练 (本地测试配置)
    # ---------------------------------------------------
    # 这些参数是为了在你现在的轻薄本 CPU 上快速跑通流程
    results = model.train(
        data=DATA_PATH,  # 你的数据集配置
        epochs=2,             # 只跑3轮，看看能不能跑通，能跑通就说明没问题
        imgsz=320,            # 这种小图跑得快
        batch=4,              # 显存/内存占用小
        device='cpu',         # 强制用 CPU
        workers=0,            # Windows 必设为 0，否则卡死
        project='NEU_Test',   # 保存结果的主文件夹
        name='ema_verification', # 实验名字
        exist_ok=True,        # 覆盖同名结果
        plots=True            # 画图
    )

if __name__ == '__main__':
    # Windows下必须放在这里面运行
    main()