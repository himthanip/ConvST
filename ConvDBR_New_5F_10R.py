import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import hamming_loss, accuracy_score 
from sklearn.metrics import f1_score, jaccard_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error
from sklearn.metrics import precision_score, recall_score

warnings.filterwarnings("ignore")

def one_error(Y_test_oe,Y_score_oe):
    print("Function Called !")
    print(Y_test_oe.shape)
    print(Y_score_oe.shape)
    
    list1=np.array(np.argmax(Y_score_oe, axis = 1)) # Find the top ranked predicted label
    print(len(list1))
    
    count_oe = 0
    
    for i in range(list1.shape[0]): # BR uses Y_test.iloc[i,list1[i]].values
        if(Y_test_oe.iloc[i, list1[i]]) == 0: # If top ranked predicted label is not in the test the count it
            count_oe += 1
    
    return count_oe / Y_score_oe.shape[0]

def compute_conviction(dset_classes, threshold = 1.2):
    X = dset_classes.values.astype(bool)
    n_samples = X.shape[0]
    class_names = np.array(dset_classes.columns)

    P = X.sum(axis = 0) / n_samples   
    B_and_not_A = (X[:, :, None] & ~X[:, None, :]).sum(axis = 0) / n_samples
    
    sup_B = X.mean(axis = 0)[:, None]
    sup_A = X.mean(axis = 0)[None, :]

    # Compute Conviction
    with np.errstate(divide='ignore', invalid='ignore'):
        conviction = (sup_B * (1 - sup_A)) / B_and_not_A

    conviction[np.isinf(conviction)] = np.nan
    np.fill_diagonal(conviction, np.nan)

    # Construct Conviction Matrix
    conviction_df = pd.DataFrame(conviction, index = class_names, columns = class_names).round(2)

    rel_lbl_lst = []
    for idx, label in enumerate(class_names):
        correlated_idx = np.where(conviction[:, idx] >= threshold)[0]
        correlated_labels = class_names[correlated_idx].tolist()
        rel_lbl_lst.append(correlated_labels)
    
    return conviction_df, rel_lbl_lst

pmac = []
pmic = []
psam = []
rmac = []
rmic = []
rsam = []
f1mac = []
f1mic = []
f1sam = []
jmac = []
jmic = []
jsam = []
ssa = []
hloss = []
rloss = []
cov = []
err_1 = []

t_tr = 0
t_pr = 0
folds = 50

for file_id in range(1, folds + 1):
    XTR_ID = 'emotions_X_Train_' + str(file_id) + ".csv"
    YTR_ID = 'emotions_Y_Train_' + str(file_id) + ".csv"
    XTST_ID = 'emotions_X_Test_' + str(file_id) + ".csv"
    YTST_ID = 'emotions_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'scenes_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'scenes_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'scenes_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'scenes_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'yeast_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'yeast_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'yeast_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'yeast_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'birds_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'birds_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'birds_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'birds_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'genbase_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'genbase_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'genbase_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'genbase_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'flags_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'flags_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'flags_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'flags_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'corel5k_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'corel5k_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'corel5k_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'corel5k_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'cal500_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'cal500_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'cal500_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'cal500_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'rcv1s1_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'rcv1s1_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'rcv1s1_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'rcv1s1_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'mediamill_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'mediamill_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'mediamill_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'mediamill_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'enron_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'enron_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'enron_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'enron_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'medical_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'medical_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'medical_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'medical_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'llog_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'llog_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'llog_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'llog_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'slashdot_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'slashdot_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'slashdot_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'slashdot_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'bibtex_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'bibtex_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'bibtex_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'bibtex_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'delicious_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'delicious_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'delicious_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'delicious_Y_Test_' + str(file_id) + ".csv"
    
    # XTR_ID = 'essays_X_Train_' + str(file_id) + ".csv"
    # YTR_ID = 'essays_Y_Train_' + str(file_id) + ".csv"
    # XTST_ID = 'essays_X_Test_' + str(file_id) + ".csv"
    # YTST_ID = 'essays_Y_Test_' + str(file_id) + ".csv"
    
    f1_xtr = "DataSet_Folds/Emotions_5Fold_10Repeats/" + XTR_ID
    f2_ytr = "DataSet_Folds/Emotions_5Fold_10Repeats/" + YTR_ID
    f3_xtst = "DataSet_Folds/Emotions_5Fold_10Repeats/" + XTST_ID
    f4_ytst = "DataSet_Folds/Emotions_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Scenes_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Scenes_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Scenes_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Scenes_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Yeast_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Yeast_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Yeast_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Yeast_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Birds_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Birds_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Birds_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Birds_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Genbase_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Genbase_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Genbase_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Genbase_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Flags_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Flags_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Flags_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Flags_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Corel5K_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Corel5K_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Corel5K_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Corel5K_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Cal500_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Cal500_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Cal500_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Cal500_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/RCV1S1_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/RCV1S1_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/RCV1S1_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/RCV1S1_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Mediamill_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Mediamill_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Mediamill_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Mediamill_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Enron_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Enron_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Enron_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Enron_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Medical_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Medical_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Medical_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Medical_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/LLog_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/LLog_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/LLog_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/LLog_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/SlashDot_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/SlashDot_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/SlashDot_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/SlashDot_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Bibtex_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Bibtex_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Bibtex_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Bibtex_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Delicious_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Delicious_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Delicious_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Delicious_5Fold_10Repeats/" + YTST_ID
    
    # f1_xtr = "DataSet_Folds/Essays_5Fold_10Repeats/" + XTR_ID
    # f2_ytr = "DataSet_Folds/Essays_5Fold_10Repeats/" + YTR_ID
    # f3_xtst = "DataSet_Folds/Essays_5Fold_10Repeats/" + XTST_ID
    # f4_ytst = "DataSet_Folds/Essays_5Fold_10Repeats/" + YTST_ID
    
    Ftr_Training = pd.read_csv(f1_xtr)
    X_Train = Ftr_Training.iloc[:, 1 : ]
    Lbl_Training = pd.read_csv(f2_ytr)
    Y_Train = Lbl_Training.iloc[:, 1 : ]
    Ftr_Testing = pd.read_csv(f3_xtst)
    X_Test = Ftr_Testing.iloc[:, 1 : ]
    Lbl_Testing = pd.read_csv(f4_ytst)
    Y_Test = Lbl_Testing.iloc[:, 1 : ]
    
    features = X_Train.columns
    classes = Y_Train.columns 
    
    conviction_matrix = np.zeros([len(classes), len(classes)])
    use_lbls = []
    i = 0
    
    cv_com_t_s = time.time()
      
    conviction_matrix, use_lbls = compute_conviction(Y_Train, threshold = 1.2)

    print("Conviction Matrix:\n", conviction_matrix)
    print("Label Wise Correlated Labels : ", use_lbls)
    
    cv_com_t_e = time.time()
    t_tr = t_tr + (cv_com_t_e - cv_com_t_s)

    print("Conviction Matrix : ")
    print(conviction_matrix)
    
    print("Labels : ", classes)
    print("Dependent Labels : ", use_lbls)
        
    Y_Pred_DFrame = pd.DataFrame()
    Y_Pred_Prob_DFrame = pd.DataFrame()
    
    pos = -1
    print("BINARY RELEVANCE", file_id)
    
    for each_label in classes:
        Y_Train_I = Y_Train[each_label]
        Y_Test_I = Y_Test[each_label]
              
        logreg = LogisticRegression()
            
        tr1_s = time.time()
        logreg.fit(X_Train, Y_Train_I)
        tr1_e = time.time()
        t_tr = t_tr + (tr1_e - tr1_s)
            
        pr1_s = time.time()
        Y_Pred_I = logreg.predict(X_Test)
        pr1_e = time.time()
        t_pr = t_pr + (pr1_e - pr1_s)
        
        Y_Pred_Prob = logreg.predict_proba(X_Test)
            
        pos = pos + 1
        Y_Pred_DFrame.insert(pos, each_label, Y_Pred_I)
        prob_label = each_label + "PROB"
        Y_Pred_Prob_DFrame.insert(pos, prob_label, Y_Pred_Prob[:, 1])

    for each_label in classes:
        X_Train[each_label] = Y_Train[each_label].values
        X_Test[each_label] = Y_Pred_DFrame[each_label].values

    Y_Pred_L2_DFrame = pd.DataFrame()
    Y_Pred_Prob_L2_DFrame = pd.DataFrame()
    pos_L2 = -1

    print("CLASSIFICATION LEVEL - 2 ", file_id)

    for each_label in classes:
        print("LABEL in CHAIN : ", each_label)
        Y_Train_L2 = pd.DataFrame()
        Y_Test_L2 = pd.DataFrame()
            
        drop_cols = []
        drop_cols.append(each_label)
        lbl_pos = list(classes).index(each_label)
        req_lbls = use_lbls[lbl_pos]
            
        for x_lbl in classes:
            if x_lbl not in req_lbls:
                drop_cols.append(x_lbl)
            
        Y_Train_L2[each_label] = Y_Train[each_label].values
        X_Train.drop(columns = drop_cols, axis = 1, inplace = True) 
        Y_Test_L2[each_label] = X_Test[each_label].values
        X_Test.drop(columns = drop_cols, axis = 1, inplace = True)
                    
        logreg_L2 = LogisticRegression()
        
        tr2_s = time.time()
        logreg_L2.fit(X_Train, Y_Train_L2)
        tr2_e = time.time()
        t_tr = t_tr + (tr2_e - tr2_s)
            
        pr2_s = time.time()
        Y_Pred_L2 = logreg_L2.predict(X_Test)
        pr2_e = time.time()
        t_pr = t_pr + (pr2_e - pr2_s)
        
        Y_Pred_Prob_L2 = logreg_L2.predict_proba(X_Test)
                      
        pos_L2 = pos_L2 + 1
        Y_Pred_L2_DFrame.insert(pos_L2, each_label, Y_Pred_L2)
        prob_label = each_label + "PROB L2"
        Y_Pred_Prob_L2_DFrame.insert(pos_L2, prob_label, Y_Pred_Prob_L2[:, 1])
        
        X_Train[each_label] = Y_Train_L2[each_label].values
        X_Test[each_label] = Y_Test_L2[each_label].values
            
        for x_lbl in drop_cols:
            X_Train[x_lbl] = Y_Train[x_lbl].values
            X_Test[x_lbl] = Y_Pred_DFrame[x_lbl].values

    pma_i = precision_score(Y_Test, Y_Pred_L2_DFrame, average = 'macro')
    pmac.append(pma_i)
    pmi_i = precision_score(Y_Test, Y_Pred_L2_DFrame, average = 'micro')
    pmic.append(pmi_i)
    pex_i = precision_score(Y_Test, Y_Pred_L2_DFrame, average = 'samples')
    psam.append(pex_i)
    
    rma_i = recall_score(Y_Test, Y_Pred_L2_DFrame, average = 'macro')
    rmac.append(rma_i)
    rmi_i = recall_score(Y_Test, Y_Pred_L2_DFrame, average = 'micro')
    rmic.append(rmi_i)
    rex_i = recall_score(Y_Test, Y_Pred_L2_DFrame, average = 'samples')
    rsam.append(rex_i)
    
    f1ma_i = f1_score(Y_Test, Y_Pred_L2_DFrame, average = 'macro')
    f1mac.append(f1ma_i)
    f1mi_i = f1_score(Y_Test, Y_Pred_L2_DFrame, average = 'micro')
    f1mic.append(f1mi_i)
    f1ex_i = f1_score(Y_Test, Y_Pred_L2_DFrame, average = 'samples')
    f1sam.append(f1ex_i)
    
    jma_i = jaccard_score(Y_Test, Y_Pred_L2_DFrame, average = 'macro')
    jmac.append(jma_i)
    jmi_i = jaccard_score(Y_Test, Y_Pred_L2_DFrame, average = 'micro')
    jmic.append(jmi_i)
    jex_i = jaccard_score(Y_Test, Y_Pred_L2_DFrame, average = 'samples')
    jsam.append(jex_i)
    
    ssa_i = accuracy_score(Y_Test, Y_Pred_L2_DFrame)
    ssa.append(ssa_i)
    hl_i = hamming_loss(Y_Test, Y_Pred_L2_DFrame)
    hloss.append(hl_i)
    
    lr_i = label_ranking_loss(Y_Test, Y_Pred_L2_DFrame)
    rloss.append(lr_i)
    cov_i = coverage_error(Y_Test, Y_Pred_L2_DFrame)
    cov.append(cov_i)
    
    err_1_i = one_error(Y_Test, Y_Pred_Prob_L2_DFrame.values)
    err_1.append(err_1_i)

print("Macro Precision : ", sum(pmac) / folds, " + / - ", np.std(pmac))
print("Micro Precision : ", sum(pmic) / folds, " + / - ", np.std(pmic))
print("Sample Precision : ", sum(psam) / folds, " + / - ", np.std(psam))

print("Macro Recall : ", sum(rmac) / folds, " + / - ", np.std(rmac))
print("Micro Recall : ", sum(rmic) / folds, " + / - ", np.std(rmic))
print("Sample Recall : ", sum(rsam) / folds, " + / - ", np.std(rsam))

print("Macro F1 Score : ", sum(f1mac) / folds, " + / - ", np.std(f1mac))
print("Micro F1 Score : ", sum(f1mic) / folds, " + / - ", np.std(f1mic))
print("Sample F1 Score : ", sum(f1sam) / folds, " + / - ", np.std(f1sam))

print("Macro Jaccard : ", sum(jmac) / folds, " + / - ", np.std(jmac))
print("Micro Jaccard : ", sum(jmic) / folds, " + / - ", np.std(jmic))
print("Sample Jaccard : ", sum(jsam) / folds, " + / - ", np.std(jsam))

print("Subset Accuracy : ", sum(ssa) / folds, " + / - ", np.std(ssa))
print("Hamming Loss : ", sum(hloss) / folds, " + / - ", np.std(hloss))

print("Ranking Loss : ", sum(rloss) / folds, " + / - ", np.std(rloss))
print("Coverage : ", sum(cov) / folds, " + / - ", np.std(cov))
print("One Error : ", sum(err_1) / folds, " + / - ", np.std(err_1))

print("Total Training Time : ", t_tr)
print("Total Preduction Tume : ", t_pr)

print("Average Training Time : ", t_tr / folds)
print("Average Prediction Time : ", t_pr / folds)

