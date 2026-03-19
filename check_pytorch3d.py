# 创建验证脚本（或直接在Python终端执行）
import torch
import pytorch3d

# 1. 打印版本信息（确认安装成功）
print("=== 版本信息 ===")
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyTorch3D 版本: {pytorch3d.__version__}")

# 2. 检查CUDA可用性（GPU环境关键）
print("\n=== CUDA 状态 ===")
print(f"PyTorch CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")

# 3. 检查PyTorch3D核心组件是否可导入
print("\n=== PyTorch3D 核心组件 ===")
try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import PerspectiveCameras, MeshRenderer, MeshRasterizer
    print("PyTorch3D 核心模块导入成功")
except ImportError as e:
    print(f"PyTorch3D 组件导入失败: {e}")