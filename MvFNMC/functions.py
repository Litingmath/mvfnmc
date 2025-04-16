
import os
from re import T
# from this import d
import numpy as np
from collections import defaultdict
import math 
import pandas as pd
from svnmc import SvNMC

def load_data_from_file(dataset, folder):

    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        next(inf)
        int_array = [line.strip("\n").split()[1:] for line in inf]
        intMat = np.array(int_array, dtype=np.float64).T    # drug-target interaction matrix
    return intMat






def kernel_RBF(X, Y, gamma):
    r2 = np.tile(np.sum(X**2, axis=1), (Y.shape[0], 1)).T + np.tile(np.sum(Y**2, axis=1), (X.shape[0], 1)) - 2 * np.dot(X, Y.T)
    k = np.exp(-r2 * gamma)
    return k

def kernel_gip_0(adjmat, dim, gamma=4):
    y = adjmat

    # Graph based kernel
    if dim == 1:
        d = kernel_RBF(y, y, gamma)
    else:
        d = kernel_RBF(y.T, y.T, gamma)

    d[d == 1] = 0
    np.fill_diagonal(d, 1)
     # 上面这个步骤对CVS=1的结果影响大!!
    return d



from scipy.spatial.distance import pdist, squareform

def kernel_to_distance(k):
    di = np.diag(k)
    d = np.tile(di, (len(k), 1)) + np.tile(di[:, np.newaxis], (1, len(k))) - 2 * k
    return d

def getGipKernel(y, gamma=0.5):
    krnl = np.dot(y, y.T)
    krnl = krnl / np.mean(np.diag(krnl))
    krnl = np.exp(-kernel_to_distance(krnl) * gamma)

  

    # krnl [krnl  == 1] = 0    # delete the sim of new drug and new target
    # np.fill_diagonal(krnl , 1)
    # 上面这个步骤对CVS=1的结果影响不大
    # krnl=process_kernel(krnl)
    # krnl=Knormailzed(krnl)

    return krnl


def process_kernel(data):
    # Make kernel matrix symmetric
    k = (data + data.T) / 2
    
    # Make kernel matrix PSD (Positive Semi-Definite)
    eigvals = np.linalg.eigvalsh(k)
    e = max(0, -np.min(eigvals) + 1e-4)
    data = k + e * np.eye(len(data))
    
    return data


def Knormailzed(K):
    # Kernel normalization
    K = np.abs(K)
    kk = K.flatten()
    kk = kk[kk != 0]
    min_v = np.min(kk)
    K[K == 0] = min_v

    D = np.diag(K)
    D = np.sqrt(D)
    S = K / (D[:, np.newaxis] * D[np.newaxis, :])

    return S

def load_sim_matrix_svnmc(data):
    
    file_path1='data-multiviews/nolinfu/'
    dfile1=os.path.join(file_path1, data +'_3_similarity_fusion_drug_k10'+ '.csv')
    dfile2=os.path.join(file_path1, data +'_3_similarity_fusion_tar_k10'+ '.csv')
    
    sdf=pd.read_csv(dfile1, header=None, dtype=np.float32).to_numpy()
   
    stf=pd.read_csv(dfile2, header=None, dtype=np.float32).to_numpy()
    
    return sdf, stf




def load_sim_matrix(data):
    
    file_path1='data-multiviews/CSV_kernels/'
    dfile1=os.path.join(file_path1, data +'_simmat_drugs_simcomp'+ '.csv')
    dfile2=os.path.join(file_path1, data +'_simmat_drugs_sider'+ '.csv')
    dfile3=os.path.join(file_path1, data +'_Drug_MACCS_fingerprint'+ '.csv')
    ds1=pd.read_csv(dfile1, index_col=None, dtype=np.float32).to_numpy()
   # In drug space, chemical structure of compounds, drug substructure fingerprint, 
   # network of drug-side effect associations, and drug known interaction profiles
    ds2=pd.read_csv(dfile2, index_col=None, dtype=np.float32).to_numpy()
   
    # admatfile=os.path.join('data-multiviews/CSV_interactions/', data +'_admat_dgc'+ '.csv')
    fp=pd.read_csv(dfile3, index_col=None, dtype=np.float32).to_numpy()
    
    ds3=kernel_gip_0(fp,1)#sim of Drug_MACCS_fingerprint e=445
    # print(ds1)
   
    
   
    Dm=np.stack((ds1, ds2, ds3), axis=0)
    for i in  np.array([0,1]):
        mat= process_kernel(Dm[i])  # Make kernel matrix PSD
        Dm[i]=Knormailzed(mat)  # Kernel normalization
    # 第三个相似矩阵是否做预处理，结果影响不大
    
    tfile1=os.path.join(file_path1, data +'_simmat_proteins_sw-n'+ '.csv')
    tfile2=os.path.join(file_path1, data +'_simmat_proteins_go'+ '.csv')
    tfile3=os.path.join(file_path1, data +'_simmat_proteins_ppi'+ '.csv')
    #In target space, protein sequence, functional annotation information,
    # protein-protein interactions network, and target known interaction profiles
    ts1=pd.read_csv(tfile1, index_col=None, dtype=np.float32).to_numpy()
   
    ts2=pd.read_csv(tfile2, index_col=None, dtype=np.float32).to_numpy()
  
    ts3=pd.read_csv(tfile3, index_col=None, dtype=np.float32).to_numpy()
   
    Tm=np.stack((ts1, ts2, ts3), axis=0)
    
    for i in np.arange(Tm.shape[0]):
        mat= process_kernel(Tm[i])
        Tm[i]=Knormailzed(mat)
    return Dm, Tm 

def spar_sim_matrix(Dm, Tm,k):
    Sm_1 = KNN_kernel_S(Dm[0], k)
    Sm_2 = KNN_kernel_S(Dm[1], k)
    Sm_3 = KNN_kernel_S(Dm[2], k)
    Sm_4 = KNN_kernel_S(Dm[3], k)
    Sm_1=(Sm_1+Sm_1.T)/2
    Sm_2=(Sm_2+Sm_2.T)/2
    Sm_3=(Sm_3+Sm_3.T)/2
    Sm_4=(Sm_4+Sm_4.T)/2

    Dm=np.stack((Sm_1, Sm_2, Sm_3, Sm_4), axis=0)

    Tm_1 = KNN_kernel_S(Tm[0], k)
    Tm_2 = KNN_kernel_S(Tm[1], k)
    Tm_3 = KNN_kernel_S(Tm[2], k)
    Tm_4 = KNN_kernel_S(Tm[3], k)
    Tm_1=(Tm_1+Tm_1.T)/2
    Tm_2=(Tm_2+Tm_2.T)/2
    Tm_3=(Tm_3+Tm_3.T)/2
    Tm_4=(Tm_4+Tm_4.T)/2

    Tm=np.stack((Tm_1, Tm_2, Tm_3, Tm_4), axis=0)
    return Dm, Tm 

def KNN_kernel_S (S, k):
    S=S- np.diag(np.diag(S))
    n = S.shape[0]
    S_knn = np.zeros([n,n])
    for i in range(n):
        sort_index = np.argsort(S[i,:])
        for j in sort_index[n-k:n]:
            if np.sum(S[i,sort_index[n-k:n]])>0:
                S_knn [i][j] = S[i][j] / (np.sum(S[i,sort_index[n-k:n]]))
    return S_knn





def get_drugs_targets_names(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        drugs_line = inf.readline().strip("\n").split()
        drugs = drugs_line[1:]  # 第一行的第一个元素是空白，所以从第二个元素开始是药物名
        targets = [line.strip("\n").split()[0] for line in inf]
    return drugs, targets


def cross_validation_svnmc(intMat, seeds, cv=0, num=10):
    cv_data = defaultdict(list)
    num_drugs, num_targets = intMat.shape
    if cv == 0:
        length=num_drugs
    if cv == 1:
        length=intMat.size
    for seed in seeds:
        prng = np.random.RandomState(seed)
        rand_ind =  prng.permutation (length)
        for i in range(num):
            if cv==1:
                test_ind = rand_ind[int(np.floor(i * length / num)):int(np.floor((i+1) * length / num))]
                test_data = np.array([[(k+1)//num_targets-1, (k+1) % num_targets-1] for k in test_ind], dtype=np.int32)
            if cv==0:
                left_out_drugs = rand_ind[int(np.floor(i * length / num)):int(np.floor((i+1) * length / num))]
                test_data = np.array([[k, j] for k in left_out_drugs for j in range(num_targets)], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W=intMat.copy()
            W[x, y] = 0
            test_idx = (x, y)  # Save test indices
            
            cv_data[seed].append((test_idx, test_data, test_label,W))
    return cv_data


def cross_validation_svnmc_all(intMat, seeds, Dm3, Tm3, cv=0, num=10):
    cv_data = defaultdict(list)
    num_drugs, num_targets = intMat.shape
    if cv == 0:
        length=num_drugs
    if cv == 1:
        length=intMat.size
    for seed in seeds:
        prng = np.random.RandomState(seed)
        rand_ind =  prng.permutation (length)
        for i in range(num):
            if cv==1:
                test_ind = rand_ind[int(np.floor(i * length / num)):int(np.floor((i+1) * length / num))]
                test_data = np.array([[(k+1)//num_targets-1, (k+1) % num_targets-1] for k in test_ind], dtype=np.int32)
            if cv==0:
                left_out_drugs = rand_ind[int(np.floor(i * length / num)):int(np.floor((i+1) * length / num))]
                test_data = np.array([[k, j] for k in left_out_drugs for j in range(num_targets)], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W=intMat.copy()
            W[x, y] = 0
            test_idx = (x, y)  # Save test indices
            Ds4=getGipKernel(W)
            Ts4=getGipKernel(W.T)
            W1=np.concatenate((Dm3, Ds4[np.newaxis, :, :]), axis=0)
            W2=np.concatenate((Tm3, Ts4[np.newaxis, :, :]), axis=0)
            
            sdf, stf=get_fusion_sim (W1,W2,10)
            
            cv_data[seed].append((test_idx, test_data, test_label,sdf, stf,W))
    return cv_data



def train_svnmc(method,model, cv_data, dataset,sdf,stf):
    aupr, auc, vec_drug, vec_tar = [], [], [], []
    print(sdf.shape)
    for seed in cv_data.keys():
        for test_idx, test_data, test_label,W in cv_data[seed]:
            # get all sim_matrix
                model.fix_model(test_idx, sdf, stf,W)
                aupr_val, auc_val = model.evaluation(test_data, test_label)
                wei_drug, wei_tar=model.return_weights()
                aupr.append(aupr_val)
                auc.append(auc_val)
                vec_drug.append(wei_drug)
                vec_tar.append(wei_tar)            
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64), np.array(vec_drug, dtype=np.float64), np.array(vec_tar, dtype=np.float64)


def train_svnmc_all(method,model, cv_data, dataset):
    aupr, auc, vec_drug, vec_tar = [], [], [], []
    # print(sdf.shape)
    for seed in cv_data.keys():
        for test_idx, test_data, test_label,sdf, stf,W in cv_data[seed]:
            # get all sim_matrix
                model.fix_model(test_idx, sdf, stf,W)
                aupr_val, auc_val = model.evaluation(test_data, test_label)
                wei_drug, wei_tar=model.return_weights()
                aupr.append(aupr_val)
                auc.append(auc_val)
                vec_drug.append(wei_drug)
                vec_tar.append(wei_tar)            
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64), np.array(vec_drug, dtype=np.float64), np.array(vec_tar, dtype=np.float64)



def cross_validation(intMat, seeds, Dm3, Tm3,cv=0, k_p=3,num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = index.size/num
        step=math.floor(step)
        for i in range(num):
            if i < num-1:
                ii = index[i*step:(i+1)*step]
            else:
                ii = index[i*step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
            elif cv == 1:
                # test_data = np.array([[k//num_targets, k % num_targets] for k in ii], dtype=np.int32)
                test_data = np.array([[(k+1)//num_targets-1, (k+1) % num_targets-1] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W=intMat.copy()
            W[x, y] = 0
            test_idx = (x, y)  # Save test indices
            Ds4=getGipKernel(W)
            Ts4=getGipKernel(W.T)
            # wei=1/Dm3.shape[0]
            # Sd=wei*Dm3[0]+wei*Dm3[1]+wei*Dm3[2]
            # St=wei*Tm3[0]+wei*Tm3[1]+wei*Tm3[2]
            # Ds4n=WKNKN(Ds4, Sd)
            # Ts4n=WKNKN(Ts4, St)
            # Ds4[x,:]=Ds4n[x,:]
            # Ds4[:,x]=Ds4n[:,x]
            # Ts4[y,:]=Ts4n[y,:]
            # Ts4[:,y]=Ts4n[:,y]
            W1=np.concatenate((Dm3, Ds4[np.newaxis, :, :]), axis=0)
            W2=np.concatenate((Tm3, Ts4[np.newaxis, :, :]), axis=0)
            
            #sdf, stf=get_fusion_sim (W1,W2,10)
            sdf=0
            stf=0

            L1=np.zeros_like(W1)
            L2=np.zeros_like(W2)
            # S1=np.zeros_like(Ds4)
            # S2=np.zeros_like(Ts4)
            for i in np.arange(W1.shape[0]):
                S1=preprocess_PNN(W1[i],k_p)
                S2=preprocess_PNN(W2[i],k_p)
                d_1 = np.sum(S1, axis=1)
                d_2= np.sum(S2, axis=1)
                D_1 = np.sqrt(d_1)
                D_2 = np.sqrt(d_2)
                with np.errstate(divide='ignore'):
                    d_inv_sqrt1 = np.diag(1.0 / D_1)
                    d_inv_sqrt2 = np.diag(1.0 / D_2)
                d_inv_sqrt1[np.isinf(d_inv_sqrt1)] = 0
                d_inv_sqrt2[np.isinf(d_inv_sqrt2)] = 0
                L1[i] = np.eye(Ds4.shape[0])-d_inv_sqrt1 @ S1 @ d_inv_sqrt1
                L2[i]= np.eye(Ts4.shape[0])-d_inv_sqrt2 @ S2 @ d_inv_sqrt2

                rows_with_zero_sum = check_row_sum_small(np.diag(d_1))
                if len(rows_with_zero_sum) > 0:
                    for row in rows_with_zero_sum:
                        print(f"药物矩阵 {i+1} 的第 {row+1} 行的和为0。")
                rows_with_zero_sum = check_row_sum_small(np.diag(d_2))
                if len(rows_with_zero_sum) > 0:
                    for row in rows_with_zero_sum:
                        print(f"靶标矩阵 {i+1} 的第 {row+1} 行的和为0。")

            cv_data[seed].append((test_idx, test_data, test_label,W,W1,W2,L1,L2,sdf, stf))
    return cv_data


def train(method,model, cv_data, dataset,Dm3,Tm3):
    aupr, auc, vec_drug, vec_tar = [], [], [], []
    # if dataset=='nr':
    #     k=5
    # else:
    #     k=10
    for seed in cv_data.keys():
        for test_idx, test_data, test_label,W,W1,W2,L1,L2,sdf, stf in cv_data[seed]:
            # get all sim_matrix
                model.fix_model(test_idx,  W1, W2,L1,L2,W,sdf, stf)
                aupr_val, auc_val = model.evaluation(test_data, test_label)
                wei_drug, wei_tar=model.return_weights()
                aupr.append(aupr_val)
                auc.append(auc_val)
                vec_drug.append(wei_drug)
                vec_tar.append(wei_tar)            
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64), np.array(vec_drug, dtype=np.float64), np.array(vec_tar, dtype=np.float64)



def preprocess_PNN(S, p):
    # Preprocess PNN sparsifies the similarity matrix S by keeping, for each
    # drug/target, the p nearest neighbors and discarding the rest.
    NN_mat1 = np.zeros_like(S)
    NN_mat2 = np.zeros_like(S)
    NN = len(NN_mat1)
    
    # For each drug/target...
    for j in range(NN):
        row = S[j, :].copy()  # Get row corresponding to current drug/target
        row[j] = 0            # Ignore self-similarity
        
        # Sort similarities descendingly and keep p NNs
        ju_list = np.sort(row)[::-1]
        ju_nearest_list_end = ju_list[p - 1]
        indx = np.where(row >= ju_nearest_list_end)[0]
        # print(indx)
        NN_mat1[j, indx] = S[j, indx]  # Keep similarities to p NNs
        NN_mat1[j, j] = S[j, j]        # Also keep the self-similarity (typically 1)
    
    for j in range(NN):
        col = S[:, j].copy()  # Get column corresponding to current drug/target
        col[j] = 0            # Ignore self-similarity
        
        # Sort similarities descendingly and keep p NNs
        ju_list = np.sort(col)[::-1]
        ju_nearest_list_end = ju_list[p - 1]
        indx = np.where(col >= ju_nearest_list_end)[0]
        # print(indx)
        NN_mat2[indx, j] = S[indx, j]  # Keep similarities to p NNs
        NN_mat2[j, j] = S[j, j]        # Also keep the self-similarity (typically 1)
    
    # Symmetrize the modified similarity matrix
    S = (NN_mat1 + NN_mat2) / 2
    
    return S




def compute_sim_matrix(cv_data,Dm3,Tm3):
    sim_matrix = defaultdict(list)
    for seed in cv_data.keys():
        for test_idx, test_data, test_label,W in cv_data[seed]:
            Ds=GIP_kernel(W) 
            # empty_rows = np.where(~W.any(axis=1))[0]  # get indices of empty rows
            # print(empty_rows)
            # empty_rows = np.where(~Ds.any(axis=1))[0]  # get indices of empty rows
            # print(empty_rows)
            Ds=new_normalization1(Ds)
            # empty_rows = np.where(~Ds.any(axis=1))[0]  # get indices of empty rows
            # print(empty_rows)
            # print(Ds)
            Ts=GIP_kernel(W.T)
            Ts=new_normalization1(Ts)
            sim_matrix[seed].append((test_idx, test_data, test_label,W,Ds,Ts))
            # wei=1/Dm3.shape[0]
            # Sd=wei*Dm3[0]+wei*Dm3[1]+wei*Dm3[2]
            # # print(Sd[0,:])
            # St=wei*Tm3[0]+wei*Tm3[1]+wei*Tm3[2]
            # Dsn=WKNKN(Ds, Sd)
            # Tsn=WKNKN(Ts, St)
            # # print(Dsn)
            # sim_matrix[seed].append((test_idx, test_data, test_label,W,Dsn,Tsn))
            # differences = np.where(Dsn != Ds)
            # if len(differences[0]) > 0:
            #     print("Differences between Ds and Dsn at indices:")
            #     for idx in zip(differences[0], differences[1]):
            #         print(f"Index {idx}: Ds = {Ds[idx]}, Dsn = {Dsn[idx]}")
            # else:
            #     print("No differences found between Ds and Dsn.")
    return sim_matrix

def WKNKN(Y, Sd, K=5, eta=0.7):
    # complete the incompleted sim-matrix 
    eta = eta**np.arange(K)
    Y=Y- np.diag(np.diag(Y))
    y2_new1 = np.zeros_like(Y)
    y2_new2 = np.zeros_like(Y)
    empty_rows = np.where(~Y.any(axis=1))[0]  # get indices of empty rows
    # print(empty_rows )
    # for each drug i...
    for i in range(len(Sd)):
        drug_sim = Sd[i, :].copy()
        drug_sim[i] = 0
        indices = np.arange(len(Sd))
        # indices1=indices.copy()
        drug_sim[empty_rows] = 0
        indices = np.delete(indices, empty_rows)
        # print("index1:", len(indices))
        indx = np.argsort(drug_sim)[::-1][:K]
        # print("index1:", indx)
        # indx = indices1[indx]
        # print("index2:", indx)
        drug_sim = Sd[i, :]
        y2_new1[i, :] = np.dot((eta * drug_sim[indx]), Y[indx, :]) / np.sum(drug_sim[indx])
        y2_new2[:,i]=y2_new1[i, :].T
       
    Y = np.maximum(Y, (y2_new1+y2_new2)/2)
    Y=(Y+Y.T)/2
    np.fill_diagonal(Y , 1)

    return Y



def svd_init(M, num_factors):
    from scipy.linalg import svd
    U, s, V = svd(M, full_matrices=False)
    ii = np.argsort(s)[::-1][:num_factors]
    s1 = np.sqrt(np.diag(s[ii]))
    U0, V0 = U[:, ii].dot(s1), s1.dot(V[ii, :])
    return U0, V0.T


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')


def load_metric_vector(file_name):
    return np.loadtxt(file_name, dtype=np.float64)


# test
# method='mvfnmc'
# dataset='nr'
# folder='datasets'
# k=10
# load_data_from_file(method,dataset, folder,k)
def get_fusion_sim (Dm,Tm,k):


    m1 = new_normalization1(Dm[0])
    m2 = new_normalization1(Dm[1])
    m3 = new_normalization1(Dm[2])
    m4 = new_normalization1(Dm[3])

    Sm_1 = KNN_kernel1(Dm[0], k)
    Sm_2 = KNN_kernel1(Dm[1], k)
    Sm_3 = KNN_kernel1(Dm[2], k)
    Sm_4 = KNN_kernel1(Dm[3], k)

    Pm = drug_updating(Sm_1,Sm_2,Sm_3,Sm_4, m1, m2,m3,m4)
    Pm_final = (Pm + Pm.T)/2
    # print(Pm_final)

    m1 = new_normalization1(Tm[0])
    m2 = new_normalization1(Tm[1])
    m3 = new_normalization1(Tm[2])
    m4 = new_normalization1(Tm[3])

    Sm_1 = KNN_kernel1(Tm[0], k)
    Sm_2 = KNN_kernel1(Tm[1], k)
    Sm_3 = KNN_kernel1(Tm[2], k)
    Sm_4 = KNN_kernel1(Tm[3], k)

    Pt = drug_updating(Sm_1,Sm_2,Sm_3,Sm_4, m1, m2,m3,m4)
    Pt_final = (Pt + Pt.T)/2
    # print(Pt_final)
    

    return Pm_final, Pt_final

def drug_updating (S1,S2,S3,S4, P1,P2,P3,P4):
    it = 0
    P = (P1+P2+P3+P4)/4
    dif = 1
    while dif>0.0000001:
        it = it + 1
        P111 =np.dot (np.dot(S1,(P2+P3+P4)/3),S1.T)
        P111 = new_normalization1(P111)
        P222 =np.dot (np.dot(S2,(P1+P3+P4)/3),S2.T)
        P222 = new_normalization1(P222)
        P333 = np.dot (np.dot(S3,(P1+P2+P4)/3),S3.T)
        P333 = new_normalization1(P333)
        P444 = np.dot(np.dot(S4,(P1+P2+P3)/3),S4.T)
        P444 = new_normalization1(P444)
        P1 = P111
        P2 = P222
        P3 = P333
        P4 = P444
        P_New = (P1+P2+P3+P4)/4
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    # print("Iter numb1", it)
    return P

def new_normalization1 (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = w[i][j]
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/((np.sum(w[i,:])-w[i,i]))
    return p


def KNN_kernel1 (S, k):
    n = S.shape[0]
    S_knn = np.zeros([n,n])
    for i in range(n):
        sort_index = np.argsort(S[i,:])
        for j in sort_index[n-k:n]:
            if np.sum(S[i,sort_index[n-k:n]])>0:
                S_knn [i][j] = S[i][j] / (np.sum(S[i,sort_index[n-k:n]]))
    return S_knn

import pickle
def save_cv_data(cv_data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(cv_data, file)

def load_cv_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)



# K = np.array([[0, 2, 3],
#               [2, 0, 4],
#               [3, 4, 0]])

# normalized_K = Knormailzed(K)
# print("Original Kernel Matrix:")
# print(K)
# print("Normalized Kernel Matrix:")
# print(normalized_K)



# import numpy as np

# # 假设有4个对称相似矩阵
# matrices = [
#     np.array([
#         [1, 2, 3],
#         [2, 0, -2],
#         [3, -2, -4]
#     ]),
#     np.array([
#         [0, 1, 1],
#         [1, -2, 1],
#         [1, 1, 0]
#     ]),
#     np.array([
#         [2, -1, -1],
#         [-1, 2, -1],
#         [-1, -1, 2]
#     ]),
#     np.array([
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]
#     ])
# ]

# def check_row_sum_zero(matrix):
#     # 返回每一行和为0的行号
#     rows_with_zero_sum = np.where(np.sum(matrix, axis=1) == 0)[0]
#     return rows_with_zero_sum

def check_row_sum_small(matrix, epsilon=1e-6):
    """
    返回每一行和足够小的行号。

    参数:
    matrix: 输入矩阵。
    epsilon: 阈值，小于此值的行和被认为是足够小。

    返回:
    一个包含每一行和足够小的行号的数组。
    """
    rows_with_small_sum = np.where(np.abs(np.sum(matrix, axis=1)) < epsilon)[0]
    return rows_with_small_sum


# for i, matrix in enumerate(matrices):
#     rows_with_zero_sum = check_row_sum_zero(matrix)
#     if len(rows_with_zero_sum) > 0:
#         for row in rows_with_zero_sum:
#             print(f"矩阵 {i+1} 的第 {row+1} 行的和为0。")