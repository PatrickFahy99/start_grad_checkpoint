import torch
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_boxplot(data, name):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=list(data.values()))
    plt.xticks(range(len(data)), list(data.keys()))
    plt.title("Boxplot of Data")
    plt.xlabel(name)
    plt.ylabel("PSNR")
    plt.show()



