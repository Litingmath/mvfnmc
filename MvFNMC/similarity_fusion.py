import numpy as np
import pandas as pd
import os
def get_fusion_sim (data, k1, k2):

    
    file_path1='data-multiviews/CSV_kernels/'
    dfile1=os.path.join(file_path1, data +'Drug_MACCS_fingerprint'+ '.csv')
    dfile2=os.path.join(file_path1, data +'simmat_drugs_sider'+ '.csv')
    dfile3=os.path.join(file_path1, data +'simmat_drugs_simcomp'+ '.csv')
    admatfile=os.path.join('data-multiviews/CSV_interactions/', data +'admat_dgc'+ '.csv')
    fp=pd.read_csv(dfile1, index_col=None, dtype=np.float32).to_numpy()
    
    ds1=GIP_kernel(fp) #sim of Drug_MACCS_fingerprint e=445
    print(ds1.shape)
    ds2=pd.read_csv(dfile2, index_col=None, dtype=np.float32).to_numpy()
    print(ds2.shape)
    ds3=pd.read_csv(dfile3, index_col=None, dtype=np.float32).to_numpy()
    print(ds3.shape)
    admat=pd.read_csv(admatfile, index_col=None, dtype=np.float32).to_numpy()
    ds4=GIP_kernel(np.transpose(admat)) 
    print(ds4.shape)

    m1 = new_normalization1(ds1)
    m2 = new_normalization1(ds2)
    m3 = new_normalization1(ds3)
    m4 = new_normalization1(ds4)

    Sm_1 = KNN_kernel1(ds1, k1)
    Sm_2 = KNN_kernel1(ds2, k1)
    Sm_3 = KNN_kernel1(ds3, k1)
    Sm_4 = KNN_kernel1(ds4, k1)

    Pm = drug_updating(Sm_1,Sm_2,Sm_3,Sm_4, m1, m2,m3,m4)
    Pm_final = (Pm + Pm.T)/2
    print(Pm_final)




    tfile1=os.path.join(file_path1, data +'simmat_proteins_go'+ '.csv')
    tfile2=os.path.join(file_path1, data +'simmat_proteins_ppi'+ '.csv')
    tfile3=os.path.join(file_path1, data +'simmat_proteins_sw-n'+ '.csv')
    ts1=pd.read_csv(tfile1, index_col=None, dtype=np.float32).to_numpy()
    print(ts1.shape)
    ts2=pd.read_csv(tfile2, index_col=None, dtype=np.float32).to_numpy()
    print(ts2.shape)
    ts3=pd.read_csv(tfile3, index_col=None, dtype=np.float32).to_numpy()
    print(ts3.shape)
    ts4=GIP_kernel(admat) 
    print(ts4.shape)

    m1 = new_normalization1(ts1)
    m2 = new_normalization1(ts2)
    m3 = new_normalization1(ts3)
    m4 = new_normalization1(ts4)

    Sm_1 = KNN_kernel1(ts1, k2)
    Sm_2 = KNN_kernel1(ts2, k2)
    Sm_3 = KNN_kernel1(ts3, k2)
    Sm_4 = KNN_kernel1(ts4, k2)

    Pt = drug_updating(Sm_1,Sm_2,Sm_3,Sm_4, m1, m2,m3,m4)
    Pt_final = (Pt + Pt.T)/2
    print(Pt_final)
    

    # sim_m1 = pd.read_csv("mydata/data/metabolite_GIP_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    # sim_m2 = pd.read_csv("mydata/data/metabolites_information_entropy_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    # sim_m3 = pd.read_csv("mydata/data/metabolites_structure_similarity.csv", index_col=0, dtype=np.float32).to_numpy()


    # sim_d1 = pd.read_csv("mydata/data/disease_semantic_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    # sim_d2 = pd.read_csv("mydata/data/disease_GIP_similarity.csv", index_col=0, dtype=np.float32).to_numpy()
    # sim_d3 = pd.read_csv("mydata/data/disease _information_entropy_similarity .csv", index_col=0, dtype=np.float32).to_numpy()


    # m1 = new_normalization1(sim_m1)
    # m2 = new_normalization1(sim_m2)
    # m3 = new_normalization1(sim_m3)

    # Sm_1 = KNN_kernel1(sim_m1, k1)
    # Sm_2 = KNN_kernel1(sim_m2, k1)
    # Sm_3 = KNN_kernel1(sim_m3, k1)

    # Pm = drug_updating(Sm_1,Sm_2,Sm_3, m1, m2,m3)
    # Pm_final = (Pm + Pm.T)/2


    # d1 = new_normalization1(sim_d1)
    # d2 = new_normalization1(sim_d2)
    # d3 = new_normalization1(sim_d3)


    # Sd_1 = KNN_kernel1(sim_d1, k2)
    # Sd_2 = KNN_kernel1(sim_d2, k2)
    # Sd_3 = KNN_kernel1(sim_d3, k2)

    # Pd = Updating1(Sd_1,Sd_2,Sd_3, d1, d2,d3)
    # Pd_final = (Pd+Pd.T)/2



    return Pm_final, Pt_final


def GIP_kernel (Asso_RNA_Dis):
    # the number of row
    nc = Asso_RNA_Dis.shape[0]
    #initate a matrix as result matrix
    matrix = np.zeros((nc, nc))
    # calculate the down part of GIP fmulate
    r = getGosiR(Asso_RNA_Dis)
    #calculate the result matrix
    for i in range(nc):
        for j in range(nc):
            #calculate the up part of GIP formulate
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i,:] - Asso_RNA_Dis[j,:]))
            if r == 0:
                matrix[i][j]=0
            elif i==j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e**(-temp_up/r)
    return matrix
def getGosiR (Asso_RNA_Dis):
# calculate the r in GOsi Kerel
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i,:])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r



def new_normalization1 (w):
    m = w.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(w[i,:])-w[i,i]>0:
                p[i][j] = w[i,j]/(2*(np.sum(w[i,:])-w[i,i]))
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



def Updating1 (S1,S2,S3, P1,P2,P3):
    it = 0
    P = (P1+P2+P3)/3
    dif = 1
    while dif>0.0000001:
        it = it + 1
        P111 =np.dot (np.dot(S1,(P2+P3)/2),S1.T)
        P111 = new_normalization1(P111)
        P222 =np.dot (np.dot(S2,(P1+P3)/2),S2.T)
        P222 = new_normalization1(P222)
        P333 = np.dot (np.dot(S3,(P1+P2)/2),S3.T)
        P333 = new_normalization1(P333)

        P1 = P111
        P2 = P222
        P3 = P333

        P_New = (P1+P2+P3)/3
        dif = np.linalg.norm(P_New-P)/np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P


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
    print("Iter numb1", it)
    return P



def GIP_kernel(association):

    nc = association.shape[0]
    matrix = np.zeros((nc, nc))
    r = getGosiR(association)
    for i in range(nc):
        for j in range(nc):
            temp_up = np.square(np.linalg.norm(association[i, :] - association[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix


def getGosiR(association):

    nc = association.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(association[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r

def sim_thresholding(matrix: np.ndarray, threshold):
    matrix_copy = matrix.copy()
    matrix_copy[matrix_copy >= threshold] = 1
    matrix_copy[matrix_copy < threshold] = 0
    print(f"rest links: {np.sum(np.sum(matrix_copy))}")
    return matrix_copy






k1=20
k2=20
data='nr_'
Pm_final, Pt_final=get_fusion_sim (data,k1, k2)
np.savetxt(data+'similarity_fusion_drug_k20.csv', Pm_final, delimiter=',')
np.savetxt(data+'similarity_fusion_tar_k20.csv', Pt_final, delimiter=',')


