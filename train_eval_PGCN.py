from opt import *

from models.PGCN import PGCN
from dataloader import *
from utils import *


if __name__ == '__main__':
    opt = OptInit().initialize()
    data_folder = "./data/{}/".format(opt.task)
    phenotype_path = os.path.join(data_folder, "Phenotype.csv")

    print('  Loading dataset ...')

    # 加载数据
    raw_features, y, nonimg, pd_dict = load_data(modal_name="MRI_CorticalThickness", data_folder=data_folder,
                                                 phenotype_path=phenotype_path)

    # 数据划分
    n_folds = 10
    cv_splits = data_split(raw_features, y, n_folds)

    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)
    sens = np.zeros(n_folds, dtype=np.float32)
    spes = np.zeros(n_folds, dtype=np.float32)

    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold))
        train_ind = cv_splits[fold][0]
        test_ind = cv_splits[fold][1]

        print('\tConstructing graph data...')
        # extract node features
        node_ftr = get_node_features(raw_features, y, train_ind)
        node_ftr = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)
        # get aff adj
        edge_index, aff_adj = get_aff_adj(node_ftr, pd_dict)
        # edge_index, aff_adj = get_PAE_inputs(node_ftr, nonimg, pd_dict)

        # build network architecture
        model = PGCN(node_ftr.shape[1], opt.num_classes, opt.dropout, hgc=opt.hgc, lg=opt.lg).to(opt.device)
        model = model.to(opt.device)

        # build loss, optimizer, metric
        weight = np.array([len(y[train_ind][np.where(y[train_ind] != 0)]) / len(train_ind),
                           len(y[train_ind][np.where(y[train_ind] != 1)]) / len(train_ind)])
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(weight).float())

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        aff_adj = torch.tensor(aff_adj, dtype=torch.float32).to(opt.device)
        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)

        def train():
            print("\tNumber of training samples %d" % len(train_ind))
            print("\tStart training...\r\n")
            acc = 0
            for epoch in range(opt.num_iter):
                model.train()
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    node_logits, edge_weights = model(node_ftr, edge_index, aff_adj)
                    loss = loss_fn(node_logits[train_ind], labels[train_ind])
                    loss.backward()
                    optimizer.step()
                correct_train, acc_train = accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])

                model.eval()
                with torch.set_grad_enabled(False):
                    node_logits, _ = model(node_ftr, edge_index, aff_adj)
                logits_test = node_logits[test_ind].detach().cpu().numpy()
                correct_test, acc_test = accuracy(logits_test, y[test_ind])
                auc_test = auc(logits_test, y[test_ind])

                print("Epoch: {},\ttrain loss: {:.4f},\ttrain acc: {:.4f}".format(epoch, loss.item(), acc_train.item()))
                if acc_test > acc and epoch > 5:
                    acc = acc_test
                    correct = correct_test
                    aucs[fold] = auc_test
                    if opt.ckpt_path != '':
                        if not os.path.exists(opt.ckpt_path):
                            os.makedirs(opt.ckpt_path)
                        torch.save(model.state_dict(), fold_model_path)

            accs[fold] = acc
            corrects[fold] = correct
            print("\r\n => Fold {} test accuacry {:.2f}%".format(fold, acc * 100))


        def evaluate(stage="test"):
            print("\tNumber of testing samples %d" % len(test_ind))
            print('\tStart testing...')
            model.load_state_dict(torch.load(fold_model_path))
            model.eval()
            node_logits, _ = model(node_ftr, edge_index, aff_adj)

            logits_test = node_logits[test_ind].detach().cpu().numpy()
            corrects[fold], accs[fold] = accuracy(logits_test, y[test_ind])
            aucs[fold] = auc(logits_test, y[test_ind])
            print("\tFold {} test accuracy {:.2f}%, AUC {:.5f}".format(fold, accs[fold] * 100, aucs[fold]))

            score_file_path = opt.ckpt_path + '/fold{}_score_{}.txt'.format(fold, stage)
            f = open(score_file_path, 'w')
            write_raw_score(f, logits_test, y[test_ind])
            matrix = [[0, 0], [0, 0]]
            matrix = matrix_sum(matrix, get_confusion_matrix(logits_test, y[test_ind]))
            print(stage + ' confusion matrix ', matrix)
            sens[fold] = get_sen(matrix)
            spes[fold] = get_spe(matrix)
            print("accuracy: {:.2f}%, sen: {:.2f}%, spe: {:.2f}%".format(get_acc(matrix) * 100, get_sen(matrix) * 100,
                                                                         get_spe(matrix) * 100))
            f.close()

        if opt.train == 1:
            train()
        elif opt.train == 0:
            evaluate()

    print("\r\n========================== Finish ==========================")
    n_samples = raw_features.shape[0]
    acc_nfold = np.sum(corrects) / n_samples
    print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
    print("=> Average test AUC in {}-fold CV: {:.5f}".format(n_folds, np.mean(aucs)))