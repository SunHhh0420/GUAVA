import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    RasterizationSettings,
    PointLights
)

# ====================== 第一步：环境检查 ======================
print("=== 1. 环境基础检查 ===")
# 设备选择（优先GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"PyTorch版本: {torch.__version__}")
try:
    import pytorch3d
    print(f"PyTorch3D版本: {pytorch3d.__version__}")
except ImportError as e:
    print(f"PyTorch3D导入失败: {e}")
    exit(1)

# 检查CUDA是否可用（GPU环境关键）
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
else:
    print("警告：使用CPU运行，渲染速度会较慢")

# ====================== 第二步：创建测试网格（立方体） ======================
print("\n=== 2. 创建测试网格（立方体） ===")
# 定义立方体的顶点（8个顶点，三维坐标）
verts = [
    [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]
]
# 定义立方体的面（12个三角形面，每个面由3个顶点索引组成）
faces = [
    [0, 1, 2], [1, 3, 2], [0, 2, 4], [2, 6, 4],
    [0, 4, 1], [4, 5, 1], [1, 5, 3], [5, 7, 3],
    [2, 3, 6], [3, 7, 6], [4, 6, 5], [6, 7, 5]
]

# 转换为PyTorch张量并移到指定设备（0.7.7版本要求batch维度必须存在）
verts = torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0)  # (1, 8, 3)
faces = torch.tensor(faces, dtype=torch.int64, device=device).unsqueeze(0)    # (1, 12, 3)

# 创建PyTorch3D的Mesh结构（核心数据结构）
try:
    mesh = Meshes(verts=verts, faces=faces)
    print("Mesh创建成功（适配0.7.7版本）")
except Exception as e:
    print(f"Mesh创建失败: {str(e)[:200]}")
    exit(1)

# ====================== 第三步：初始化渲染器（纯0.7.7原生API） ======================
print("\n=== 3. 初始化渲染器 ===")
# 1. 相机设置（移除所有新版参数，仅保留核心必选参数）
cameras = PerspectiveCameras(
    device=device,
    focal_length=torch.tensor([[1.0, 1.0]], device=device),  # (1, 2)
    principal_point=torch.tensor([[0.0, 0.0]], device=device),  # (1, 2)
    R=torch.eye(3, device=device).unsqueeze(0),  # (1, 3, 3)
    T=torch.tensor([[0.0, 0.0, 3.0]], device=device)  # (1, 3)
)

# 2. 光栅化设置（仅保留0.7.7支持的参数）
raster_settings = RasterizationSettings(
    image_size=256,          # 输出图像分辨率 256x256
    blur_radius=0.0,         # 模糊半径（0为无模糊）
    faces_per_pixel=1        # 每个像素保留的面数
)

# 3. 光源设置（移除batch_size，仅保留核心参数）
lights = PointLights(
    device=device,
    location=torch.tensor([[0.0, 0.0, -3.0]], device=device)  # (1, 3)
)

# 4. 构建渲染器（0.7.7版本原生写法）
rasterizer = MeshRasterizer(
    cameras=cameras,
    raster_settings=raster_settings
)
# 0.7.7版本：HardPhongShader必须传入cameras和lights
shader = HardPhongShader(
    device=device,
    cameras=cameras,
    lights=lights
)
renderer = MeshRenderer(
    rasterizer=rasterizer,
    shader=shader
)

# ====================== 第四步：执行渲染（核心功能测试） ======================
print("\n=== 4. 执行网格渲染 ===")
try:
    # 0.7.7版本：渲染时仅传入mesh，相机已在光栅化器中绑定
    images = renderer(mesh)
    print(f"渲染成功！输出图像形状: {images.shape}")
    print(f"图像数据类型: {images.dtype}")
    print(f"图像像素值范围: [{images.min().item():.4f}, {images.max().item():.4f}]")
    
    # 可选：保存渲染结果到本地
    try:
        from PIL import Image
        img_np = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img_np).save("pytorch3d_render_test.png")
        print("渲染结果已保存为: pytorch3d_render_test.png")
    except ImportError:
        print("提示：未安装pillow，跳过图片保存（不影响渲染验证）")
    
    print("\n✅ PyTorch3D 网格渲染功能验证通过！")
except Exception as e:
    print(f"\n❌ 渲染失败，错误信息: {str(e)[:200]}")
    print("\n⚠️  建议：降级PyTorch到1.13.1+cu118（与0.7.7完美兼容）")
    print("   执行命令：pip uninstall -y torch torchvision torchaudio && pip install torch==1.13.1+cu118 torchvision==0.14.1+cu118 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu118")