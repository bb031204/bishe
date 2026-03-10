"""
手动生成超图可视化
用于训练后查看超图结构
"""
import os
import sys
import argparse
import numpy as np
import torch

# 添加src到路径
sys.path.insert(0, os.path.dirname(__file__))

from src.graph.hypergraph_utils import visualize_hypergraph


def main():
    parser = argparse.ArgumentParser(description="Visualize Hypergraph from Cache")
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='data/cache',
        help='Cache directory containing hypergraph .npz files'
    )
    parser.add_argument(
        '--element',
        type=str,
        default='Temperature',
        help='Element name (Temperature, Cloud, Humidity, Wind)'
    )
    parser.add_argument(
        '--K',
        type=int,
        default=5,
        help='Top-K value used in hypergraph construction'
    )
    parser.add_argument(
        '--similarity',
        type=str,
        default='euclidean',
        help='Similarity metric for semantic hypergraph'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hypergraph Visualization Tool")
    print("=" * 60)
    print()
    
    # 构建文件路径
    nei_cache_file = os.path.join(
        args.cache_dir,
        f"{args.element}_nei_K{args.K}.npz"
    )
    
    sem_cache_file = os.path.join(
        args.cache_dir,
        f"{args.element}_sem_K{args.K}_{args.similarity}.npz"
    )
    
    # 检查文件是否存在
    if not os.path.exists(nei_cache_file):
        print(f"❌ Neighbourhood hypergraph cache not found: {nei_cache_file}")
        print("   Please run training first to generate hypergraph cache.")
        return
    
    if not os.path.exists(sem_cache_file):
        print(f"❌ Semantic hypergraph cache not found: {sem_cache_file}")
        print("   Please run training first to generate hypergraph cache.")
        return
    
    # 加载邻域超图
    print("Loading neighbourhood hypergraph...")
    nei_data = np.load(nei_cache_file)
    H_nei = torch.from_numpy(nei_data['H']).float()
    print(f"✓ Neighbourhood hypergraph loaded: {H_nei.shape}")
    
    # 加载语义超图
    print("Loading semantic hypergraph...")
    sem_data = np.load(sem_cache_file)
    H_sem = torch.from_numpy(sem_data['H']).float()
    print(f"✓ Semantic hypergraph loaded: {H_sem.shape}")
    print()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成可视化
    print("Generating visualizations...")
    print()
    
    # 邻域超图
    nei_output = os.path.join(
        args.output_dir,
        f"hypergraph_neighbourhood_{args.element}_K{args.K}.png"
    )
    print(f"Visualizing neighbourhood hypergraph...")
    visualize_hypergraph(H_nei, save_path=nei_output)
    print(f"✓ Saved: {nei_output}")
    print()
    
    # 语义超图
    sem_output = os.path.join(
        args.output_dir,
        f"hypergraph_semantic_{args.element}_K{args.K}_{args.similarity}.png"
    )
    print(f"Visualizing semantic hypergraph...")
    visualize_hypergraph(H_sem, save_path=sem_output)
    print(f"✓ Saved: {sem_output}")
    print()
    
    print("=" * 60)
    print("✓ Visualization completed!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  1. {nei_output}")
    print(f"  2. {sem_output}")
    print()
    print("You can now view these images with your image viewer.")


if __name__ == "__main__":
    main()
