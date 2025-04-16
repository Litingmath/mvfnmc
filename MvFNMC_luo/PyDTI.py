
import os
import sys
import time
import getopt
from mvfnmc1new import MvFNMC1new
from mvfnmc1old import MvFNMC1old
import cv_eval
from functions import *
from nrlmf import NRLMF
from svnmc import SvNMC
from mvfnmc1 import MvFNMC1
from mvfnmc1new1 import MvFNMC1new1
from mvfnmc2 import MvFNMC2
import pickle

from new_pairs import novel_prediction_analysis


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "m:d:f:c:s:o:n:p", ["method=", "dataset=", "data-dir=", "cvs=", "specify-arg=", "method-options=", "predict-num=", "output-dir="])
        # print("opts=", opts)
        # print("args=", args)
    except getopt.GetoptError:
        sys.exit()

   

    cvs, sp_arg, model_settings, predict_num = 1, 1, [], 0

    # seeds = [7771, 8367, 22, 1812, 4659]
    # seeds = np.random.choice(10000, 5, replace=False)
    seeds = [7771]

    for opt, arg in opts:
        if opt == "--method":
            method = arg
        if opt == "--dataset":
            dataset = arg
        if opt == "--data-dir":
            data_dir = arg
        if opt == "--output-dir":
            output_dir = arg
        if opt == "--cvs":
            cvs = int(arg)
        if opt == "--specify-arg":
            sp_arg = int(arg)
        if opt == "--method-options":
            model_settings = [s.split('=') for s in str(arg).split()]
        if opt == "--predict-num":
            predict_num = int(arg)

    output_dir = os.path.join(os.path.pardir, 'luoout/output_AvgWei')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
   
    if method== 'mvfnmc1new1':
        if dataset == 'nr':
            args = {'lambda_d':2, 'alpha':1, 'gamma_d':1.9, 'gamma_t':1.1, 'max_iter':18, 'k_p':4, 'K':5, 'eta':0.7, 'pre':1, 'delay':5}
            ker_p=3
        if dataset == 'ic':
            args = {'lambda_d':1, 'alpha':2**-4, 'gamma_d':1.2, 'gamma_t':1.1, 'max_iter':10, 'k_p':4, 'K':8, 'eta':0.8, 'pre':1, 'delay':1}
            ker_p=int(args['k_p'])
        if dataset == 'gpcr':  
            args = {'lambda_d':2, 'alpha':2**-4, 'gamma_d':1.5, 'gamma_t':1.1, 'max_iter':9, 'k_p':5, 'K':6, 'eta':0.7, 'pre':1, 'delay':1}
            ker_p=int(args['k_p'])
        if dataset == 'e':  
            args = {'lambda_d':4, 'alpha':2**-5, 'gamma_d':1.3, 'gamma_t':1.1, 'max_iter':10, 'k_p':3, 'K':10, 'eta':0.9, 'pre':1, 'delay':1}
            ker_p=int(args['k_p'])
        if dataset == 'luo':  
            args = {'lambda_d':4, 'alpha':2**-5, 'gamma_d':1.1, 'gamma_t':1.1, 'max_iter':10, 'k_p':3, 'K':10, 'eta':0.9, 'pre':1, 'delay':1}
            ker_p=int(args['k_p'])

    if method== 'mvfnmc2':
        if dataset == 'nr':
            args = {'lam':0.125, 'lambda_d':2, 'alpha':1, 'gamma_d':2.2, 'gamma_t':1.1, 'max_iter':20, 'k_p':4, 'K':5, 'eta':0.7, 'pre':1, 'delay':5}
            ker_p=3
        if dataset == 'ic':
            args = {'lam':0.5,'lambda_d':4, 'alpha':2**-5, 'gamma_d':1.1, 'gamma_t':1.1, 'max_iter':13, 'k_p':4, 'K':8, 'eta':0.8, 'pre':1, 'delay':1}
            ker_p=int(args['k_p'])
        if dataset == 'gpcr':  
            args = {'lam':0.5, 'lambda_d':4, 'alpha':2**-4, 'gamma_d':2.0, 'gamma_t':1.1, 'max_iter':12, 'k_p':5, 'K':6, 'eta':0.7, 'pre':1, 'delay':1}
            ker_p=int(args['k_p'])
        if dataset == 'e':  
            args = {'lam':2, 'lambda_d':4, 'alpha':2**-5, 'gamma_d':1.1, 'gamma_t':1.1, 'max_iter':5, 'k_p':3, 'K':10, 'eta':0.9, 'pre':1, 'delay':1}
            ker_p=int(args['k_p'])
            
        if dataset == 'luo':  
            args = {'lam':2, 'lambda_d':4, 'alpha':2**-5, 'gamma_d':1.1, 'gamma_t':1.1, 'max_iter':10, 'k_p':5, 'K':10, 'eta':0.9, 'pre':1, 'delay':1}
            ker_p=int(args['k_p'])




    
    
    for key, val in model_settings:
        args[key] = val

   
    datasetfile="luo_admat_dgc.txt"
    intMat = load_data_from_file('luo/',datasetfile)   # drug-target interaction matrix
    drug_names, target_names = get_drugs_targets_names(dataset, 'luo')
    Dm3,Tm3=load_sim_matrix(dataset)   # original sim_matrices
    

    if predict_num == 0:
        if cvs == 1:  # CV setting CVS1
            X, cv, D, T= intMat,  1, Dm3,Tm3
        if cvs == 2:  # CV setting CVS2
            X, cv, D, T = intMat, 0, Dm3,Tm3
        if cvs == 3:  # CV setting CVS3
            X,  cv, D, T= intMat.T,  0, Tm3,Dm3
          
        
        ker_p=3
        
        cv_data =  cross_validation_lcq (X, seeds, D, T, cv,ker_p)
        print("finished")
        



    
    if sp_arg == 1 or predict_num > 0:
        tic = time.clock()
        if method== 'mvfnmc1new1':
            model = MvFNMC1new1(lambda_d=args['lambda_d'],  alpha=args['alpha'], gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], max_iter=args['max_iter'], k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=args['delay'], cvs=cvs)
        if method== 'mvfnmc2':
            model = MvFNMC2(lam=args['lam'], lambda_d=args['lambda_d'],  alpha=args['alpha'], gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], max_iter=args['max_iter'], k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=args['delay'],cvs=cvs)
       
                  

         
        


        cmd = str(model)
        if predict_num == 0:
            print ("Dataset:"+dataset+" CVS:"+str(cvs)+"\n"+cmd)
            aupr_vec, auc_vec,vec_drug, vec_tar = train(method,model,  cv_data, dataset, D, T)
            # print(vec_drug.shape)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            nd=vec_drug.shape[1]
            nt=vec_tar.shape[1]
            drug_avg=np.zeros(nd)
            tar_avg=np.zeros(nt)
            for i in range(nd):
                drug_avg[i],drug_conf=mean_confidence_interval(vec_drug[:,i])
            for i in range(nt):
                tar_avg[i],tar_conf=mean_confidence_interval(vec_tar[:,i])
            print ("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
            # Print the weights for the drug view
            print("weights for drug view: ", end="")
            for weight in drug_avg:
                print("%.6f" % weight, end=" ")
            print()
            # Print the weights for the target view
            print("weights for target view: ", end="")
            for weight in tar_avg:
                print("%.6f" % weight, end=" ")
            print()
            write_metric_vector_to_file(auc_vec, os.path.join(output_dir, method+"_auc_cvs"+str(cvs)+"_"+dataset+".txt"))
            write_metric_vector_to_file(aupr_vec, os.path.join(output_dir, method+"_aupr_cvs"+str(cvs)+"_"+dataset+".txt"))
        elif predict_num > 0:
            print ("Dataset:"+dataset+"\n"+cmd)
            seed = 7771 if method == 'cmf' else 22
            model.fix_model(intMat, intMat, drugMat, targetMat, seed)
            x, y = np.where(intMat == 0)
            scores = model.predict_scores(zip(x, y), 5)
            ii = np.argsort(scores)[::-1]
            predict_pairs = [(drug_names[x[i]], target_names[y[i]], scores[i]) for i in ii[:predict_num]]
            new_dti_file = os.path.join(output_dir, "_".join([method, dataset, "new_dti.txt"]))
            novel_prediction_analysis(predict_pairs, new_dti_file, 'database')

if __name__ == "__main__":
    main(sys.argv[1:])
