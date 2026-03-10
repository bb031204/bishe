"""
环境检查脚本
检查所有依赖是否正确安装
"""
import sys

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ⚠️  警告: Python版本过低，建议使用3.8+")
        return False
    return True


def check_package(package_name, import_name=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: 未安装")
        return False


def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA: 可用")
            print(f"  - CUDA版本: {torch.version.cuda}")
            print(f"  - GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print(f"✗ CUDA: 不可用 (将使用CPU)")
            return False
    except:
        return False


def check_data_paths():
    """检查数据路径"""
    import os
    import yaml
    
    try:
        with open('configs/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        paths = [
            config['data']['train_path'],
            config['data']['val_path'],
            config['data']['test_path'],
            config['data']['position_path']
        ]
        
        all_exist = True
        for path in paths:
            if os.path.exists(path):
                print(f"✓ 数据文件存在: {path}")
            else:
                print(f"✗ 数据文件不存在: {path}")
                all_exist = False
        
        return all_exist
    
    except Exception as e:
        print(f"✗ 检查数据路径失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("HyperGKAN 环境检查")
    print("=" * 60)
    print()
    
    # Python版本
    print("【1/4】检查Python版本...")
    python_ok = check_python_version()
    print()
    
    # 依赖包
    print("【2/4】检查依赖包...")
    packages = [
        ('torch', 'torch'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scipy', 'scipy'),
        ('sklearn', 'sklearn'),
        ('yaml', 'yaml'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
        ('pykan', 'kan')  # 可选
    ]
    
    packages_ok = []
    for pkg_name, import_name in packages:
        ok = check_package(pkg_name, import_name)
        packages_ok.append(ok)
    
    print()
    
    # CUDA
    print("【3/4】检查CUDA...")
    cuda_ok = check_cuda()
    print()
    
    # 数据文件
    print("【4/4】检查数据文件...")
    data_ok = check_data_paths()
    print()
    
    # 总结
    print("=" * 60)
    print("检查总结")
    print("=" * 60)
    
    critical_packages = packages_ok[:7]  # 前7个是必需的
    pykan_ok = packages_ok[-1] if len(packages_ok) > 7 else False
    
    if python_ok and all(critical_packages):
        print("✓ 核心环境: 正常")
    else:
        print("✗ 核心环境: 有问题，请安装缺失的包")
        print("  运行: pip install -r requirements.txt")
    
    if pykan_ok:
        print("✓ pykan: 已安装 (将使用KAN)")
    else:
        print("⚠️  pykan: 未安装 (将自动降级为MLP)")
        print("  如需使用KAN，运行: pip install pykan")
    
    if cuda_ok:
        print("✓ CUDA: 可用 (将使用GPU加速)")
    else:
        print("⚠️  CUDA: 不可用 (将使用CPU)")
    
    if data_ok:
        print("✓ 数据文件: 都存在")
    else:
        print("✗ 数据文件: 有缺失，请检查config.yaml中的路径")
    
    print()
    
    if python_ok and all(critical_packages) and data_ok:
        print("🎉 环境配置完成，可以开始训练！")
        print()
        print("运行以下命令开始训练:")
        print("  python train.py")
    else:
        print("❌ 环境配置未完成，请解决上述问题")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
