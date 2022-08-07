import matplotlib.pyplot as plt
from utils import *
from opt import *

if __name__ == '__main__':
    opt = OptInit().initialize()
    path = opt.ckpt_path
    n_fold = 10
    matrix = [[0, 0], [0, 0]]
    preds = []
    labels = []
    for i in range(n_fold):
        with open(path + '/fold{}_score_test.txt'.format(i), 'r') as f:
            file = f.readlines()
            for j in range(len(file)):
                data = file[j].split("__")
                tmp = [data[0], data[1]]
                tmp = np.array(tmp).astype(np.float32)
                preds.append(tmp)
                label = data[2].split()[0]
                labels.append(round(float(label)))
        f.close()
    print(torch.tensor(labels))
    matrix = matrix_sum(matrix, get_confusion_matrix(preds, labels))
    acc = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1])
    sen = matrix[1][1] / (matrix[0][1] + matrix[1][1])  # 敏感性
    spe = matrix[0][0] / (matrix[0][0] + matrix[1][0])
    PPV = (matrix[0][0]) / (matrix[0][0] + matrix[0][1])
    f1 = 2 / (1 / spe + 1 / PPV)
    auc = auc(preds, labels)
    print(matrix)

    print("acc: {:.4}%".format(acc*100))

    print("sen: {:.4}%".format(sen*100))

    print("spe: {:.4}%".format(spe*100))

    print("F1: {:.4}%".format(f1*100))

    print("auc: {:.4}".format(auc))

    # 加标题
    plt.title('Results of CN vs. MCI classification')
    # 柱状图
    plt.bar(0, acc, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='ACC')
    plt.bar(1.5, sen, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='SEN')
    plt.bar(3, spe, error_kw={'ecolor': '0.2', 'capsize': 6}, alpha=0.7, label='SPE')
    # 横轴
    plt.xticks([0, 1.5, 3], ['ACC', 'SEN', 'SPE'])
    # 图例
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    plt.tight_layout()
    # 保存
    #plt.savefig(path + "/res.png")
    # 展示
    #plt.show()
