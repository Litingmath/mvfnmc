'''
Multi-views fused nonnegative matrix completion methods for  drug-target interactions prediction with a fused vieww: SvNMF
'''

from ast import Return
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

class SvNMC2:

    def __init__(self, lam=0.5, lambda_d=2.5,  alpha=1, gamma_d=1.1, gamma_t=1.1, max_iter=10, k_p=3, K=5, eta=0.7, pre=1, delay=10):
        self.lam =float(lam)
        self.lambda_d = float(lambda_d)
        self.alpha = float(alpha)
        self.gamma_d=float(gamma_d)
        self.gamma_t=float(gamma_t)
        self.max_iter = int(max_iter)
        self.k_p=int(k_p)
        self.K=int(K)
        self.eta=float(eta)
        self.pre=int(pre)
        self.delay=int(delay)

    def return_weights(self):
            return self.wei_drug, self.wei_tar

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

    
    
    def PLiBCD(self,test_idx,W1,W2,y):
        num_1, num_2 = y.shape
        lambda1=self.lambda_d
        lambda2 = lambda1 * num_2 ** 2 / num_1 ** 2
        ala1=self.alpha
        ala2=ala1 * num_2 ** 2 / num_1 ** 2
        self.nd=4
        self.wei_drug=np.full((self.nd,),1/self.nd)
        self.wei_tar=np.full((self.nd,),1/self.nd)
        
        
        if  self.pre == 1:
            Z_new = self.preprocess_Y(y, W1,W2)
            self.Z = y.copy()
            self.Z[test_idx] = Z_new[test_idx]
        else:
            self.Z=y.copy()

        Fd = self.Z.copy()
        Ft = self.Z.copy()


        if self.max_iter==0:
            # print ("predicting DTIS by WKNKN")
            pass
        else:
            L1= self.compute_normalized_laplacian(W1)
            L2= self.compute_normalized_laplacian(W2)
            for i in range(self.max_iter):
                AA = lambda1 * L1 + (1 + ala1) * np.eye(num_1)
                gradFd = np.dot(AA, Fd) - self.Z
                lc = np.linalg.norm(AA, 'fro')
                LA = Fd - gradFd / lc
                Fd = np.maximum(0, LA)
              
                BB = lambda2 * L2 + (self.lam + ala2) * np.eye(num_2)
            
                gradFt = np.dot(Ft, BB) - self.lam*self.Z
                lc =  np.linalg.norm(BB, 'fro')
                LA = Ft - gradFt / lc
                Ft= np.maximum(0, LA)
                
                self.Z = y.copy()
                self.Z[test_idx] = (Fd[test_idx]+self.lam*Ft[test_idx])/(1+self.lam)
 
               
    


    def compute_normalized_laplacian(self, W):
    
        D = np.diag(np.sum(W, axis=1))
    
        # 计算标准拉普拉斯矩阵 L
        L = D - W
    
        # 计算度矩阵的逆平方根
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))
    
        # 计算标准化的拉普拉斯矩阵 L_norm
        L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
        return L_norm


    def fix_model(self, test_idx,W1,W2,y):
     
        self.PLiBCD(test_idx,W1,W2,y)


    def log_likelihood(self):
        loglik = 0
        A = np.dot(self.U, self.V.T)
        B = A * self.intMat
        loglik += np.sum(B)
        A = np.exp(A)
        A += self.ones
        A = np.log(A)
        A = self.intMat1 * A
        loglik -= np.sum(A)
        loglik -= 0.5 * self.lambda_d * np.sum(np.square(self.U))+0.5 * self.lambda_t * np.sum(np.square(self.V))
        loglik -= 0.5 * self.alpha * np.sum(np.diag((np.dot(self.U.T, self.DL)).dot(self.U)))
        loglik -= 0.5 * self.beta * np.sum(np.diag((np.dot(self.V.T, self.TL)).dot(self.V)))
        return loglik

    def construct_neighborhood(self, drugMat, targetMat):
        self.dsMat = drugMat - np.diag(np.diag(drugMat))
        self.tsMat = targetMat - np.diag(np.diag(targetMat))
        if self.K1 > 0:
            S1 = self.get_nearest_neighbors(self.dsMat, self.K1)
            self.DL = self.laplacian_matrix(S1)
            S2 = self.get_nearest_neighbors(self.tsMat, self.K1)
            self.TL = self.laplacian_matrix(S2)
        else:
            self.DL = self.laplacian_matrix(self.dsMat)
            self.TL = self.laplacian_matrix(self.tsMat)

    
    def get_nearest_neighbors(self, S, size=5):
        m, n = S.shape
        X = np.zeros((m, n))
        for i in range(m):
            ii = np.argsort(S[i, :])[::-1][:min(size, n)]
            X[i, ii] = S[i, ii]
        return X

    
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

    def pass_self_to_function(self):
        init_Z=self.Z
        return init_Z
    

    def __str__(self):
        return "Model: SvNMC,   lambda_d:%s,  gamma_d:%s, gamma_t:%s,alpha:%s, max_iter:%s, k_p:%s, K:%s, eta:%s, pre:%s, delay:%s" % (self.lambda_d,  self.gamma_d, self.gamma_t, self.alpha, self.max_iter, self.k_p, self.K, self.eta, self.pre, self.delay)
