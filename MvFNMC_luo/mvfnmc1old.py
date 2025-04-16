'''
Multi-views fused nonnegative matrix completion methods for  drug-target interactions prediction 
'''

from ast import Return
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from scipy.linalg import solve_sylvester

class MvFNMC1old:

    def __init__(self, lambda_d=2.5,  alpha=1, gamma=1.1, max_iter=10, K=5, eta=0.7, pre=1):
        self.lambda_d = float(lambda_d)
        self.alpha = float(alpha)
        self.gamma=float(gamma)
        self.max_iter = int(max_iter)
        self.K=int(K)
        self.eta=float(eta)
        self.pre=int(pre)

  
    def calculate_weights(self, F, Ld, Lt):
        
        w1=np.zeros(self.nd)
        w2=np.zeros(self.nd)
        e=1/(self.gamma-1)
        for i in range(self.nd):
            r_u_d = np.trace(np.dot(F.T, np.dot(Ld[i], F)))
            w1[i]=(1/r_u_d)**e
            r_v_t = np.trace(np.dot(F, np.dot(Lt[i], F.T)))
            w2[i]=(1/r_v_t)**e
        
       
        for i in range(self.nd):
            self.wei_drug[i]=w1[i]/np.sum(w1)
    
            self.wei_tar[i]=w2[i]/np.sum(w2)
    




    def  combine_kernels(self,Ld,Lt):
        dn=Ld.shape[1]
        tn=Lt.shape[1]
        L1=np.zeros((dn,dn))
        L2=np.zeros((tn,tn))
        # print(L1.shape)
        for i in range(self.nd):
            L1 +=(self.wei_drug[i]**self.gamma)*Ld[i]
            L2 +=(self.wei_tar[i]**self.gamma)*Lt[i]
        return L1, L2

    
    def preprocess_Y(self, Y, Sd, St):

        K=self.K
        
        eta = self.eta ** np.arange(K)
        
        y2_new1 = np.zeros(Y.shape)
        y2_new2 = np.zeros(Y.shape)
        
        empty_rows = np.where(np.all(Y == 0, axis=1))[0]  # indices of empty rows
        empty_cols = np.where(np.all(Y == 0, axis=0))[0]  # indices of empty columns
        
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
        
        for j in range(len(St)):
            target_sim = St[j, :].copy()
            target_sim[j] = 0
            indices = np.arange(len(St))
            target_sim[empty_cols] = 0
            indices = np.delete(indices, empty_cols)
            
            indx = np.argsort(target_sim)[::-1][:K]
            # indx = indices[indx]
            
            target_sim = St[j, :]
            y2_new2[:, j] = np.dot(Y[:, indx], (eta * target_sim[indx])) / np.sum(target_sim[indx])
        
        Y_new=np.maximum(Y, (y2_new1 + y2_new2) / 2)
        return Y_new
        
    # def preprocess_Y(self, Y, Sd, St):

    #     K=self.K
        
    #     eta = self.eta ** np.arange(K)
        
    #     y2_new1 = np.zeros(Y.shape)
    #     y2_new2 = np.zeros(Y.shape)
        
    #     empty_rows = np.where(np.all(Y == 0, axis=1))[0]  # indices of empty rows
    #     empty_cols = np.where(np.all(Y == 0, axis=0))[0]  # indices of empty columns
        
    #     for i in range(len(Sd)):
    #         drug_sim = Sd[i, :].copy()
    #         drug_sim[i] = 0
    #         indices = np.arange(len(Sd))
    #         non_empty_indices = np.setdiff1d(indices, empty_rows)  # ignore empty rows
    #         # print(non_empty_indices)

    #         drug_sim_filtered = drug_sim[non_empty_indices]
    #         indx = np.argsort(drug_sim_filtered)[::-1]  # sort descendingly
    #         indx = indx[:K]  # keep only similarities of K nearest neighbors
    #         nearest_indices = non_empty_indices[indx]  # and their indices
    #         drug_sim = Sd[i, :]
    #         weighted_sim = eta * drug_sim[nearest_indices]
    #         y2_new1[i, :] = np.dot(weighted_sim, Y[nearest_indices, :]) / np.sum(drug_sim[nearest_indices])
              
    #     for j in range(len(St)):
    #         target_sim = St[j, :].copy()
    #         target_sim[j] = 0
    #         indices = np.arange(len(St))
    #         non_empty_indices = np.setdiff1d(indices, empty_cols) 
    #         tar_sim_filtered = target_sim[non_empty_indices]
    #         indx = np.argsort(tar_sim_filtered)[::-1]  # sort descendingly
    #         indx = indx[:K]  # keep only similarities of K nearest neighbors
    #         nearest_indices = non_empty_indices[indx]  # and their indices
            
    #         target_sim = St[j, :]
    #         y2_new2[:, j] = np.dot(Y[:, indx], (eta * target_sim[nearest_indices])) / np.sum(target_sim[nearest_indices])
        
    #     Y_new=np.maximum(Y, (y2_new1 + y2_new2) / 2)
    #     return Y_new

    
    

    def PLiBCD(self,test_idx,W1,W2,L1,L2,y):
      
        num_1, num_2 = y.shape
        lambda1=self.lambda_d
        lambda2 = lambda1 * num_2 ** 2 / num_1 ** 2
        self.nd=4
        Ld=np.zeros((self.nd, num_1, num_1))
        Lt=np.zeros((self.nd, num_2, num_2))
        self.wei_drug=np.full((self.nd,),1/self.nd)
        self.wei_tar=np.full((self.nd,),1/self.nd)
        Sd=self.wei_drug[0]*(W1[0]+W1[1]+W1[2]+W1[3])
        St=self.wei_tar[0]*(W2[0]+W2[1]+W2[2]+W2[3])

       
        Z_new = self.preprocess_Y(y, Sd,St)
        self.Z=Z_new
        # self.Z=y.copy()
        # self.Z[test_idx]=Z_new[test_idx]
        LapA = self.Z.copy()  
        
        
        # for i in range(self.nd):
        #    L1= self.compute_normalized_laplacian(W1[i])
        #    Ld[i]=L1
        #    L2= self.compute_normalized_laplacian(W2[i])
        #    Lt[i]=L2 
        Ld=L1
        Lt=L2

           
        for i in range(self.max_iter):


            L1, L2=self.combine_kernels(Ld,Lt)
            
            if num_1 <= num_2:
                AA = lambda1 * L1 + (1 + self.alpha) * np.eye(num_1)
                BB = lambda2 * L2
            else:
                AA = lambda1 * L1
                BB = lambda2 * L2 + (1 + self.alpha) * np.eye(num_2)
            
            gradF = np.dot(AA, LapA) + np.dot(LapA, BB) - self.Z
            lc = np.linalg.norm(AA, 'fro') + np.linalg.norm(BB, 'fro')
            # lc = np.linalg.norm(AA, 2) + np.linalg.norm(BB, 2)
            LA = LapA - gradF / lc
            LapA = np.maximum(0, LA)
            
            self.Z = y.copy()
            self.Z[test_idx] = LapA[test_idx]   
            self.calculate_weights(LapA, Ld, Lt)
           
     
        if  self.pre == 1:
            Sd=self.wei_drug[0]*W1[0]+self.wei_drug[1]*W1[1]+self.wei_drug[2]*W1[2]+self.wei_drug[3]*W1[3]
            St=self.wei_tar[0]*W2[0]+self.wei_tar[1]*W2[1]+self.wei_tar[2]*W2[2]+self.wei_tar[3]*W2[3]
            Z_new = self.preprocess_Y(y, Sd,St)
            zn=y.copy()
            zn[test_idx]  = Z_new [test_idx] 
           
            self.Z=np.maximum(zn, self.Z)
        else:
            # print("No preprocesses")
            pass

    
    def MvLapRLP(self,W1,W2,L1,L2,y):
      
        num_1, num_2 = y.shape
        lambda1=self.lambda_d
        lambda2 = lambda1 * num_2 ** 2 / num_1 ** 2
        self.nd=4
        Ld=np.zeros((self.nd, num_1, num_1))
        Lt=np.zeros((self.nd, num_2, num_2))
        self.wei_drug=np.full((self.nd,),1/self.nd)
        self.wei_tar=np.full((self.nd,),1/self.nd)
        Sd=self.wei_drug[0]*(W1[0]+W1[1]+W1[2]+W1[3])
        St=self.wei_tar[0]*(W2[0]+W2[1]+W2[2]+W2[3])

       
        Z_new = self.preprocess_Y(y, Sd,St)
        self.Z=Z_new
       
        Ld=L1
        Lt=L2

           
        for i in range(self.max_iter):


            L1, L2=self.combine_kernels(Ld,Lt)


            AA=lambda1 * L1+np.eye(num_1)
            BB=lambda2 * L2
            CC=Z_new
            LapA=solve_sylvester(AA,BB,CC)
            self.Z=LapA
            
            
            self.calculate_weights(LapA, Ld, Lt)
           
     
        

    
    def compute_normalized_laplacian(self, W):
    
        D = np.diag(np.sum(W, axis=1))
    
        # 计算标准拉普拉斯矩阵 L
        L = D - W
    
        # 计算度矩阵的逆平方根, W存在全为0的值
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)+1e-8))
    
        # 计算标准化的拉普拉斯矩阵 L_norm
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
        return L_norm


    def fix_model(self, test_idx,W1,W2,L1,L2,y):
     
        self.PLiBCD(test_idx,W1,W2,L1,L2,y)
        # self.PLiBCD_L2(test_idx,W1,W2,y,InitZ)
        # self.MvLapRLP(W1,W2,L1,L2,y)


    

    
    
    
    def predict_scores(self, test_data, N):
        dinx = np.array(list(self.train_drugs))
        DS = self.dsMat[:, dinx]
        tinx = np.array(list(self.train_targets))
        TS = self.tsMat[:, tinx]
        scores = []
        for d, t in test_data:
            if d in self.train_drugs:
                if t in self.train_targets:
                    val = np.sum(self.U[d, :]*self.V[t, :])
                else:
                    jj = np.argsort(TS[t, :])[::-1][:N]
                    val = np.sum(self.U[d, :]*np.dot(TS[t, jj], self.V[tinx[jj], :]))/np.sum(TS[t, jj])
            else:
                if t in self.train_targets:
                    ii = np.argsort(DS[d, :])[::-1][:N]
                    val = np.sum(np.dot(DS[d, ii], self.U[dinx[ii], :])*self.V[t, :])/np.sum(DS[d, ii])
                else:
                    ii = np.argsort(DS[d, :])[::-1][:N]
                    jj = np.argsort(TS[t, :])[::-1][:N]
                    v1 = DS[d, ii].dot(self.U[dinx[ii], :])/np.sum(DS[d, ii])
                    v2 = TS[t, jj].dot(self.V[tinx[jj], :])/np.sum(TS[t, jj])
                    val = np.sum(v1*v2)
            scores.append(np.exp(val)/(1+np.exp(val)))
        return np.array(scores)
    def evaluation(self, test_data, test_label):
        x, y = test_data[:, 0], test_data[:, 1]
        scores=self.Z[x,y]
        prec, rec, thr = precision_recall_curve(test_label, scores)
        aupr_val = auc(rec, prec)
        fpr, tpr, thr = roc_curve(test_label, scores)
        auc_val = auc(fpr, tpr)
        return aupr_val, auc_val
    
    def GIP_kernel (self,Asso_RNA_Dis):
    # the number of row
        nc = Asso_RNA_Dis.shape[0]
        #initate a matrix as result matrix
        matrix = np.zeros((nc, nc))
        # calculate the down part of GIP fmulate
        r = self.getGosiR(Asso_RNA_Dis)
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
        matrix=(matrix+matrix.T)/2
        return matrix
    def getGosiR (self,Asso_RNA_Dis):
        # calculate the r in GOsi Kerel
        nc = Asso_RNA_Dis.shape[0]
        summ = 0
        for i in range(nc):
            x_norm = np.linalg.norm(Asso_RNA_Dis[i,:])
            x_norm = np.square(x_norm)
            summ = summ + x_norm
        r = summ / nc
        return r
    



   
    def __str__(self):
        return "Model: MvFNMC1old, lambda_d:%s,  gamma:%s, alpha:%s, max_iter:%s, K:%s, eta:%s, pre:%s" % (self.lambda_d,  self.gamma, self.alpha, self.max_iter, self.K, self.eta, self.pre)
