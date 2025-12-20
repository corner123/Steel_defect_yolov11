# export_test.py
from ultralytics import YOLO

# 1. 指向你的配置文件 (确保文件名对)
model = YOLO("yolov11n_ema.yaml")

# 2. 打印一下网络结构，看看有没有 EMA 层
# 如果这里没报错，说明 block.py 和 tasks.py 改对了
model.info()

# 3. 运行导出
print("正在尝试导出 ONNX...")
model.export(format="onnx")
print("✅ 导出成功！")