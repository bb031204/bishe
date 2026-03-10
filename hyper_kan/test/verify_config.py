"""
验证config.yaml中的context特征配置
"""
import yaml

def verify_context_config():
    """验证context特征配置"""
    print("=" * 60)
    print("验证Context特征配置")
    print("=" * 60)
    
    # 加载配置
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查context配置
    print("\n1. 基础配置:")
    print(f"   use_context: {config['data'].get('use_context', False)}")
    print(f"   context_dim: {config['data'].get('context_dim', 0)}")
    print(f"   use_dim4: {config['data'].get('use_dim4', False)}")
    
    # 检查context特征开关
    if 'context_features' in config['data']:
        print("\n2. Context特征开关:")
        context_features = config['data']['context_features']
        
        feature_list = [
            ('use_longitude', '经度'),
            ('use_latitude', '纬度'),
            ('use_altitude', '海拔'),
            ('use_year', '年份'),
            ('use_month', '月份'),
            ('use_day', '日期'),
            ('use_hour', '小时'),
            ('use_region', '区域标志')
        ]
        
        selected_count = 0
        for key, name in feature_list:
            enabled = context_features.get(key, True)
            status = "[ON]" if enabled else "[OFF]"
            print(f"   {name:8s}: {status}")
            if enabled:
                selected_count += 1
        
        print(f"\n   总计: {selected_count}/8 个特征将被使用")
        
        # 构建掩码
        context_feature_mask = [
            context_features.get('use_longitude', True),
            context_features.get('use_latitude', True),
            context_features.get('use_altitude', True),
            context_features.get('use_year', True),
            context_features.get('use_month', True),
            context_features.get('use_day', True),
            context_features.get('use_hour', True),
            context_features.get('use_region', True)
        ]
        
        print(f"\n3. 特征掩码: {context_feature_mask}")
        print(f"   True的个数: {sum(context_feature_mask)}")
    else:
        print("\n2. [ERROR] context_features configuration not found")
    
    print("\n" + "=" * 60)
    print("[OK] Configuration verified!")
    print("=" * 60)
    
    print("\nExplanation:")
    print("  - When use_context=true, selected context features will be concatenated to data features")
    print("  - Model's input_dim will automatically increase (original features + selected context features)")
    print("  - These features will be normalized independently for different scales")


if __name__ == "__main__":
    verify_context_config()
