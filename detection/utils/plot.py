import seaborn as sns
import matplotlib.pyplot as plt


def plothist(axe, data, cat_names, title=" ", rotation=90):
    cat_len = len(cat_names)
    sns.histplot(data, ax=axe, bins=cat_len, discrete=True)
    axe.set_xticks(range(0, cat_len))
    axe.set_xticklabels(cat_names, rotation=rotation)
    axe.set_title(title)
    axe.bar_label(axe.containers[1])