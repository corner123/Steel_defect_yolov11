from ultralytics import YOLO

DATA_PATH = r"NEU-DET/steel_defect.yaml"
def main():
    # 1. 加载模型
    # yolo11n.pt 会自动下载，如果下载慢，可以去官网下载放同目录下
    model = YOLO('yolo11n.pt')

    # 2. 开始训练
    # 针对轻薄本的关键参数配置：
    results = model.train(
        data=DATA_PATH,  # 指定数据集配置
        epochs=2,  # 训练轮数，轻薄本建议先跑50-100轮看效果
        imgsz=640,  # 图片大小，显存不够改 416
        batch=16,  # 批次大小。显存 4G 改 8，6G/8G 可以试 16
        device='cpu',  # 使用 GPU 0
        workers=2,  # Windows下PyCharm设多线程容易报错，建议 0 或 2
        project='NEU_Result',  # 结果保存的主目录
        name='exp1',  # 实验名称
        amp=True,  # 开启混合精度训练，省显存且速度快
        exist_ok=True  # 覆盖同名实验文件夹
    )

    # 3. 简单验证一下
    # 使用验证集的第一张图做测试
    results = model.val()


if __name__ == '__main__':
    # Windows下必须放在 if __name__ == '__main__': 下运行
    main()