import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
import gensim.models.keyedvectors as word2vec
import pdb

import LIBLINEAR.mlc_NE_LIBLINEAR as mlc_NE_LIBLINEAR
import LIBLINEAR.mcc_liblinear as mcc_liblinear


def evaluate(model, sess, args, epoch):
    # save embeddings and test
    if args.normalized:
        rep = sess.run(model.get_normalized_embeddings())
    else:
        rep = sess.run(model.embedding_T)
    sio.savemat(args.rep, {'rep': rep})

    if args.task=='mcc':
        results = mcc_liblinear.mcc_liblinear_one_file(args)
        save_results(epoch, results, './result/{}-mcc-{}-{}.txt'.format(args.dataset, args.normalized, args.base))
        print_info(epoch, results)

    elif args.task=='mlc':
        results = mlc_NE_LIBLINEAR.MLC(args)
        save_results(epoch, results, './result/{}-mlc-{}-{}.txt'.format(args.dataset, args.normalized, args.base))
        print_info(epoch, results)

    elif args.task=='re':
        train_adj = sio.loadmat(args.input_net)['network']
        re_check_index = [int(i) for i in args.check_reconstruction.split(',')]
        results = check_reconstruction(rep, train_adj, re_check_index)
        save_results(epoch, results, './result/{}-re.txt'.format(args.dataset))
        print_info(epoch, results)

    elif args.task=='mcc+re':
        results = mcc_liblinear.mcc_liblinear_one_file(args)
        save_results(epoch, results, './result/{}-mcc.txt'.format(args.dataset))
        print_info(epoch, results)

        train_adj = sio.loadmat(args.input_net)['network']
        re_check_index = [int(i) for i in args.check_reconstruction.split(',')]
        results = check_reconstruction(rep, train_adj, re_check_index)
        save_results(epoch, results, './result/{}-re.txt'.format(args.dataset))
        print_info(epoch, results)

    elif args.task=='mlc+re':
        results = mlc_NE_LIBLINEAR.MLC(args)
        save_results(epoch, results, './result/{}-mlc.txt'.format(args.dataset))
        print_info(epoch, results)

        train_adj = sio.loadmat(args.input_net)['network']
        re_check_index = [int(i) for i in args.check_reconstruction.split(',')]
        results = check_reconstruction(rep, train_adj, re_check_index)
        save_results(epoch, results, './result/{}-re.txt'.format(args.dataset))
        print_info(epoch, results)

    elif args.task=='lp':
        # train_adj = sio.loadmat(args.input_net)['network']
        # original_adj = sio.loadmat(args.original_graph)['network']
        # lp_check_index = [int(i) for i in args.check_link_prediction.split(',')]
        # results = check_link_prediction(rep, train_adj, original_adj, lp_check_index)
        # save_results(epoch, results, './result/{}-lp.txt'.format(args.dataset))
        # print_info(epoch, results)
        repMatToTxt(rep, args.lp_path+args.rep_txt)
        results = linkPred_AUC(args.lp_path+args.rep_txt, args.lp_path+args.train_pos_file, args.lp_path+args.train_neg_file, \
                                 args.lp_path+args.test_pos_file, args.lp_path+args.test_neg_file)
        save_results(epoch, results, './result/{}-lp-AUC-{}-{}.txt'.format(args.dataset, args.normalized, args.base))
        print_info(epoch, results)

    return results


def print_info(epoch, results):
    for i in range(results.shape[0]):
        info = 'Epoch {} |'.format(epoch)
        for j in range(results.shape[1]):
            info = info + ' {:.4f} |'.format(results[i, j])
        print(info)

def save_results(epoch, results, file):
    f = open(file, 'a')
    N, D = results.shape[0], results.shape[1]
    f.write
    for i in range(N):
        f.write('{:3d}\t'.format(epoch))
        for n in range(D):
            f.write('{:.4f}\t'.format(results[i,n]))
            if n==(D-1): f.write('\n')
    f.close()

def print_settings(args, flag='settings', best_epoch=None):
    if args.task=='mcc':
        _print_settings(args, './result/{}-mcc-{}-{}.txt'.format(args.dataset, args.normalized, args.base), flag=flag, best_epoch=best_epoch)
    elif args.task=='mlc':
        _print_settings(args, './result/{}-mlc-{}-{}.txt'.format(args.dataset, args.normalized, args.base), flag=flag, best_epoch=best_epoch)
    elif args.task=='re':
        _print_settings(args, './result/{}-re.txt'.format(args.dataset), flag=flag, best_epoch=best_epoch)
    elif args.task=='mcc+re':
        _print_settings(args, './result/{}-mcc.txt'.format(args.dataset), flag=flag, best_epoch=best_epoch)
        _print_settings(args, './result/{}-re.txt'.format(args.dataset), flag=flag, best_epoch=best_epoch)
    elif args.task=='mlc+re':
        _print_settings(args, './result/{}-mlc.txt'.format(args.dataset), flag=flag, best_epoch=best_epoch)
        _print_settings(args, './result/{}-re.txt'.format(args.dataset), flag=flag, best_epoch=best_epoch)
    elif args.task=='lp':
        _print_settings(args, './result/{}-lp-AUC-{}-{}.txt'.format(args.dataset, args.normalized, args.base), flag=flag, best_epoch=best_epoch)

def _print_settings(args, file, flag='settings', best_epoch=None):
    resultFile = open(file, 'a')
    if flag=='settings':
        resultFile.write('===========================================================\n')
        settings = 'input {}|adv {}|adver {}|eps {}|embed_size {}|nepoch {}|batch_size {}|lr {}|reg_adv {}|walk_l {}'\
                   .format(args.input_net, args.adv, args.adver, args.eps, args.embed_size, args.nepoch, args.batch_size, \
                           args.lr, args.reg_adv, args.walk_length) + \
                   '|num_w {}|w_s {}|p {}|q {}|K {}|normalized {}|adapt_l2 {}|model {}|order {}\n' \
                   .format(args.num_walks, args.window_size, args.p, args.q, args.negative, args.normalized, args.adapt_l2, args.base, args.order)
        resultFile.write(settings)
        resultFile.write('-----------------------------------------------------------\n')
    elif flag=='breakline':
        resultFile.write('-----------------------------------------------------------\n')
    elif flag=='best_epoch':
        resultFile.write('-----------------------------------------------------------\n')
        resultFile.write('Best Epoch: {}\n'.format(best_epoch))
        resultFile.write('-----------------------------------------------------------\n')

    resultFile.close()

def getSimilarity(result):
    print("getting similarity...")
    return np.dot(result, result.T)
    
def check_reconstruction(embedding, adj, check_index):
    def get_precisionK(embedding, adj, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        for ind in sortedInd:
            x = ind / adj.shape[0]
            y = ind % adj.shape[0]
            count += 1
            if (adj[x].toarray()[0][y] == 1 or x == y):
                cur += 1 
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
        
    precisionK = get_precisionK(embedding, adj, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.4f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return np.reshape(np.array(ret), (1, -1))

def check_link_prediction(embedding, train_adj, origin_adj, check_index):
    def get_precisionK(embedding, train_adj, origin_adj, max_index):
        print("get precisionK...")
        similarity = getSimilarity(embedding).reshape(-1)
        sortedInd = np.argsort(similarity)
        cur = 0
        count = 0
        precisionK = []
        sortedInd = sortedInd[::-1]
        N = train_adj.shape[0]
        for ind in sortedInd:
            x = ind / N
            y = ind % N
            if (x == y or train_adj[x].toarray()[0][y] == 1):
                continue 
            count += 1
            if (origin_adj[x].toarray()[0][y] == 1):
                cur += 1
            precisionK.append(1.0 * cur / count)
            if count > max_index:
                break
        return precisionK
    precisionK = get_precisionK(embedding, train_adj, origin_adj, np.max(check_index))
    ret = []
    for index in check_index:
        print("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
        ret.append(precisionK[index - 1])
    return np.reshape(np.array(ret), (1, -1))
 

def check_multi_label_classification(X, Y, test_ratio = 0.9):
    def small_trick(y_test, y_pred):
        y_pred_new = np.zeros(y_pred.shape,np.bool)
        sort_index = np.flip(np.argsort(y_pred, axis = 1), 1)
        for i in range(y_test.shape[0]):
            num = sum(y_test[i])
            for j in range(num):
                y_pred_new[i][sort_index[i][j]] = True
        return y_pred_new
        
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_ratio)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    y_pred = clf.predict_proba(x_test)
    
    ## small trick : we assume that we know how many label to predict
    y_pred = small_trick(y_test, y_pred)
    
    micro = f1_score(y_test, y_pred, average = "micro")
    macro = f1_score(y_test, y_pred, average = "macro")
    return "micro_f1: %.4f macro_f1 : %.4f" % (micro, macro)
    #############################################


######################
# For Link Prediction
######################

def repMatToTxt(rep_mat, rep_file):
    '''
    Input Example:
        rep_mat = '/home/wonniu/Embeddings/Link-Prediction/RNE/cora-50-rep.mat'
        rep_file = '/home/wonniu/Embeddings/Link-Prediction/RNE/cora-50-rep.txt'
    '''
    fileID = open(rep_file, 'w')
    node_num, dim = rep_mat.shape[0], rep_mat.shape[1]
    fileID.write('{:d} {:d}\n'.format(node_num, dim))
    for i in range(node_num):
        fileID.write('{:d} '.format(i+1))
        for j in range(dim):
            if(j!=dim-1):
                fileID.write('{:f} '.format(rep_mat[i, j]))
            else:
                fileID.write('{:f}\n'.format(rep_mat[i, j]))
    fileID.close()


def linkPred_AUC(rep_file, train_pos_file, train_neg_file, test_pos_file, test_neg_file):
    '''
    Input Example:
        rep_file = '/home/wonniu/Embeddings/Link-Prediction/RNE/cora-50-rep.txt'
        train_pos_file = '/home/wonniu/Embeddings/Link-Prediction/AUC/cora-50-train-pos.net'
        train_neg_file = '/home/wonniu/Embeddings/Link-Prediction/AUC/cora-50-train-neg.net'
        test_pos_file = '/home/wonniu/Embeddings/Link-Prediction/AUC/cora-50-test-pos.net'
        test_neg_file = '/home/wonniu/Embeddings/Link-Prediction/AUC/cora-50-test-neg-'
    '''
    auc = np.zeros((1, 11))
    for i in range(1, 11):
        model = word2vec.KeyedVectors.load_word2vec_format(rep_file, binary=False)
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        X_test_pos = []
        X_test_neg = []
        with open (train_pos_file,'rb') as f:
            for line in f:
                pair = [s.decode('utf-8') for s in line.strip().split()]
                if pair[0] not in model.vocab or pair[1] not in model.vocab:
                    continue
                else:
                    head = model[pair[0]]
                    tail = model[pair[1]]
                    feature = np.multiply(head,tail)
                    X_train.append(feature)
                    Y_train.append(1)

        with open (train_neg_file,'rb') as f:
            for line in f:
                pair = [s.decode('utf-8') for s in line.strip().split()]
                if pair[0] not in model.vocab or pair[1] not in model.vocab:
                    continue
                else:
                    head = model[pair[0]]
                    tail = model[pair[1]]
                    feature = np.multiply(head,tail)
                    X_train.append(feature)
                    Y_train.append(-1)

        clf = linear_model.SGDClassifier(loss='squared_hinge')

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # print('----------------------------------')
        # print(X_train.shape, Y_train.shape)
        # print('----------------------------------')
        clf.fit(X_train,Y_train)

        Y_unknown = []
        Y_score_unknown = []
        with open (test_pos_file,'rb') as f:
            for line in f:
                pair = [s.decode('utf-8') for s in line.strip().split()]
                if pair[0] not in model.vocab or pair[1] not in model.vocab:
                    Y_unknown.append(1)
                    Y_score_unknown.append(0.5)
                else:
                    head = model[pair[0]]
                    tail = model[pair[1]]
                    feature = np.multiply(head,tail)
                    X_test.append(feature)
                    X_test_pos.append(feature)
                    Y_test.append(1)

        with open (test_neg_file + str(i) + '.net','rb') as f:
            for line in f:
                pair = [s.decode('utf-8') for s in line.strip().split()]
                if pair[0] not in model.vocab or pair[1] not in model.vocab:
                    Y_unknown.append(-1)
                    Y_score_unknown.append(0.5)
                else:
                    head = model[pair[0]]
                    tail = model[pair[1]]
                    feature = np.multiply(head,tail)
                    X_test.append(feature)
                    X_test_neg.append(feature)
                    Y_test.append(-1)

        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        Y_pred = clf.decision_function(X_test)
        Y_pred_pos = clf.decision_function(X_test_pos)
        Y_pred_neg = clf.decision_function(X_test_neg)
        Y_pred = np.clip(Y_pred, -50, 50)
        for p in range(len(Y_pred)):
            Y_pred[p] = 1 / (1 + np.exp(-Y_pred[p]))

        tmp = roc_auc_score(Y_test, Y_pred)
        auc[0, i-1] = tmp

    auc[0, 10] = np.sum(auc, axis=-1)/10
    return auc
    
