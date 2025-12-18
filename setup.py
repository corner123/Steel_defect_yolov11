from setuptools import setup, find_packages

setup(
    name='ultralytics',
    version='8.3.0',
    description='Ultralytics YOLOv8 modes',
    packages=find_packages(),
    # 自动识别当前目录下的 ultralytics 文件夹作为包
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: AGPL-3.0 License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'matplotlib>=3.3.0',
        'numpy>=1.22.2',
        'opencv-python>=4.6.0',
        'pillow>=7.1.2',
        'pyyaml>=5.3.1',
        'requests>=2.23.0',
        'scipy>=1.4.1',
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'tqdm>=4.64.0',
        'pandas>=1.1.4',
        'seaborn>=0.11.0',
        'psutil',
        'py-cpuinfo',
    ],
)