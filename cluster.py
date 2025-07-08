import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data(embedding_path, types_path):
    """
    加载保存的嵌入向量和标签数据
    Args:
        embedding_path (str): 嵌入向量文件路径 (.npy)
        types_path (str): 标签文件路径 (.txt 或 .csv)
    Returns:
        latent (np.ndarray): 嵌入向量
        true_label (np.ndarray): 真实标签
        pred_label (np.ndarray): 预测标签
    """
    latent = np.load(embedding_path)
    types_df = pd.read_csv(types_path)
    true_label = types_df['True'].values
    pred_label = types_df['Pred'].values
    return latent, true_label, pred_label


def visualize_clusters(latent, true_label, pred_label, save_path, epoch):
    """
    使用 UMAP 降维并可视化聚类结果
    Args:
        latent (np.ndarray): 嵌入向量
        true_label (np.ndarray): 真实标签
        pred_label (np.ndarray): 预测标签
        save_path (str): 保存图像的路径
        epoch (int): 训练轮数（用于文件名）
    """
    # 使用 UMAP 进行降维
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(latent)

    # 绘制散点图
    plt.figure(figsize=(12, 5))

    # 真实标签散点图
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=true_label, palette='tab10', s=10)
    plt.title('True Labels')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # 预测标签散点图
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=pred_label, palette='tab10', s=10)
    plt.title('Predicted Labels')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    plt.tight_layout()
    # 保存图像
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}/clustering_visualization_epoch_{epoch}.png", dpi=300)
    plt.close()


def main():
    # 设置文件路径
    dataset = "Pollen"  # 替换为你的数据集名称
    epoch = 80  # 替换为你要分析的轮数
    base_path = f"D:/python learning/project1/scMAE_MSA1.1-main/data/data_output/{dataset}"
    embedding_path = f"{base_path}/embedding_{epoch}.npy"
    types_path = f"{base_path}/types_{epoch}.csv"
    save_path = f"{base_path}/visualization"

    # 加载数据
    print(f"Loading data from {embedding_path} and {types_path}...")
    latent, true_label, pred_label = load_data(embedding_path, types_path)

    # 可视化聚类结果
    print("Visualizing clustering results...")
    visualize_clusters(latent, true_label, pred_label, save_path, epoch)
    print(f"Visualization saved to {save_path}/clustering_visualization_epoch_{epoch}.png")


if __name__ == "__main__":
    main()
