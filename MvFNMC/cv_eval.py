
from ast import Return
import time
from functions import *
from nrlmf import NRLMF
from svnmc import SvNMC
from mvfnmc1 import MvFNMC1
from mvfnmc1old import MvFNMC1old
# from mvfnmc1new import MvFNMC1new
from mvfnmc1new1 import MvFNMC1new1
from mvfnmc2 import MvFNMC2
# from netlaprls import NetLapRLS
# from blm import BLMNII
# from wnngip import WNNGIP
# from kbmf import KBMF
# from cmf import CMF

import os

# directory = "/path/to/your"
# filename = "file.txt"
# file_path = os.path.join(directory, filename)

# write_string_to_file(file_path, content)
def write_string_to_file( content, file_path):
    """
    将字符串写入指定路径的文件。

    参数:
    file_path (str): 文件路径。
    content (str): 要写入文件的字符串。
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content+"\n")
        print(f"内容已成功写入 {file_path}")
    except IOError as e:
        print(f"写入文件时出错: {e}")

# # 示例用法
# file_path = '/path/to/your/file.txt'  # 替换为你的文件路径
# content = "这是要写入文件的字符串内容。"

# write_string_to_file(file_path, content)

def cv_eval_other2(output_dir,method, dataset, cv_data, cvs, args,Dm3,Tm3):
    max_aupr, aupr_opt, results_list = 0, [], []
    # if method=='mvfnmc1':
        # output_dir = os.path.join(os.path.pardir, 'out/outputsvnmc_lam')
    for x in  np.arange(-3, 2): # [2**-2, 2**-1, 2**0, 2**1, 2**2]:
        for  y in np.arange(-2, 3):
            for z in np.arange(-5, 2):
                tic = time.clock()
                
                if method== 'mvfnmc2':
                    model = MvFNMC2(lam=float(2)**(x), lambda_d=float(2)**(y),  alpha=float(2)**(z), gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], max_iter=args['max_iter'], k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=args['delay'])

                
                cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                # print (cmd)
                aupr_vec, auc_vec, vec_drug, vec_tar = train(method,model, cv_data, dataset,Dm3,Tm3)
                #method,model, cv_data, dataset,Dm3,Tm3
                aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                cmd+="\n"+"auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic)
                results_list.append(cmd)
            
                if aupr_avg > max_aupr:
                    max_aupr = aupr_avg
                    aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
                    opt_para=[float(2)**(x), float(2)**(y), float(2)**(z)]
    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    # cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print (cmd)
    results_list.append(cmd)
    out_file=os.path.join(output_dir, method+"_opt_other"+str(cvs)+"_"+dataset+".txt")
    with open(out_file, "w", encoding="utf-8") as file:
    # 循环遍历字符串列表，将每个字符串写入文件
        for string in results_list:
            file.write(string + "\n")  # 每行结束后添加换行符

    return opt_para


def cv_eval_other(output_dir,method, dataset, cv_data, cvs, args,Dm3,Tm3):
    max_aupr, aupr_opt, results_list = 0, [], []
    # if method=='mvfnmc1':
        # output_dir = os.path.join(os.path.pardir, 'out/outputsvnmc_lam')
    for x in  np.arange(-2, 3): # [2**-2, 2**-1, 2**0, 2**1, 2**2]:
        for  y in np.arange(-5, 2):
            for z in np.arange(5,6):
                tic = time.clock()
                
                if method== 'mvfnmc1new1':
                    model = MvFNMC1new1(lambda_d=float(2)**(x),  alpha=float(2)**(y), gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], max_iter=z, k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=args['delay'])
                if method== 'mvfnmc2':
                    model = MvFNMC2(lam=args['lam'], lambda_d=args['lambda_d'],  alpha=args['alpha'], gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], max_iter=args['max_iter'], k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=args['delay'])

                
                cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                # print (cmd)
                aupr_vec, auc_vec,vec_drug, vec_tar = train(method,model, cv_data, dataset,Dm3,Tm3)
                #method,model, cv_data, dataset,Dm3,Tm3
                aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                cmd+="\n"+"auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic)
                results_list.append(cmd)
            
                if aupr_avg > max_aupr:
                    max_aupr = aupr_avg
                    aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
                    opt_para=[float(2)**(x), float(2)**(y), z]
    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    # cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print (cmd)
    results_list.append(cmd)
    out_file=os.path.join(output_dir, method+"_opt_other"+str(cvs)+"_"+dataset+".txt")
    with open(out_file, "w", encoding="utf-8") as file:
    # 循环遍历字符串列表，将每个字符串写入文件
        for string in results_list:
            file.write(string + "\n")  # 每行结束后添加换行符

    return opt_para

def cv_eval_para_new(output_dir,method, dataset, cv_data, cvs, args,Dm3,Tm3):
    max_aupr, aupr_opt, results_list = 0, [], []
    
    # x_range= np.arange(1.1, 3.1, 0.1)
    x_range=np.arange(1,11)
    re_aupr=[0]*len(x_range)
    re_auc=[0]*len(x_range)
    i=0
    for x in x_range:
        tic = time.clock()
        if method== 'mvfnmc1new1':
            model = MvFNMC1new1(lambda_d=args['lambda_d'],  alpha=args['alpha'], gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], max_iter=args['max_iter'], k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=x)
        if method== 'mvfnmc2':
            model = MvFNMC2(lam=args['lam'], lambda_d=args['lambda_d'],  alpha=args['alpha'], gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], max_iter=args['max_iter'], k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=x)

        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
        # print (cmd)
        aupr_vec, auc_vec,vec_drug, vec_tar= train(method,model, cv_data, dataset,Dm3,Tm3)
        #method,model, cv_data, dataset,Dm3,Tm3
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        re_aupr[i]=aupr_avg
        re_auc[i]=auc_avg
        i +=1
        cmd+="\n"+"auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic)
        results_list.append(cmd)  
        if aupr_avg > max_aupr:
            max_aupr = aupr_avg
            aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
            opt_para=float(2)**(x)
    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    # cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print (cmd)
    results_list.append(cmd)
    out_file=os.path.join(output_dir, method+"_opt_lamd"+str(cvs)+"_"+dataset+".txt")
    with open(out_file, "w", encoding="utf-8") as file:
    # 循环遍历字符串列表，将每个字符串写入文件
        for string in results_list:
            file.write(string + "\n")  # 每行结束后添加换行符

    return opt_para,re_aupr,re_auc




def cv_eval_iter(output_dir,method, dataset, cv_data, cvs, args,Dm3,Tm3):
    max_aupr, aupr_opt, results_list = 0, [], []
    
    iter_range=np.arange(1,21,2)
    re_aupr=[0]*len(iter_range)
    re_auc=[0]*len(iter_range)
    for i in np.arange(len(iter_range)):
        tic = time.clock()
        if method=='svnmc':
            model = SvNMC(lambda_d=args['lambda_d'],  alpha=args['alpha'], max_iter=iter_range[i], K=args['K'], eta=args['eta'], pre=args['pre'])
        if method== 'mvfnmc1new1':
            model = MvFNMC1new1(lambda_d=args['lambda_d'],  alpha=args['alpha'], gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], max_iter=iter_range[i], k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=args['delay'])
        if method== 'mvfnmc2':
            model = MvFNMC2(lam=args['lam'], lambda_d=args['lambda_d'],  alpha=args['alpha'], gamma_d=args['gamma_d'], gamma_t=args['gamma_t'], max_iter=iter_range[i], k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=args['delay'])

        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
        # print (cmd)
        aupr_vec, auc_vec,vec_drug, vec_tar= train(method,model, cv_data, dataset,Dm3,Tm3)
        #method,model, cv_data, dataset,Dm3,Tm3
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        re_aupr[i]=aupr_avg
        re_auc[i]=auc_avg
        cmd+="\n"+"auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic)
        results_list.append(cmd)  
        if aupr_avg > max_aupr:
            max_aupr = aupr_avg
            aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
            opt_iter=iter
    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    # cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print (cmd)
    results_list.append(cmd)
    out_file=os.path.join(output_dir, method+"_opt_iter"+str(cvs)+"_"+dataset+".txt")
    with open(out_file, "w", encoding="utf-8") as file:
    # 循环遍历字符串列表，将每个字符串写入文件
        for string in results_list:
            file.write(string + "\n")  # 每行结束后添加换行符

    return opt_iter,re_aupr,re_auc


def cv_eval_paras(output_dir,method, dataset, cv_data, cvs, args,Dm3,Tm3):
    max_aupr, aupr_opt, results_list = 0, [], []
    # if method=='mvfnmc1':
        # output_dir = os.path.join(os.path.pardir, 'out/outputsvnmc_lam')
    for gam1 in  np.arange(1.1, 2.5, 0.1):
        for gam2 in  np.arange(1.1, 2.5, 0.1):
            tic = time.clock()
            # if method=='mvfnmc1': 
            #     model = MvFNMC1(lambda_d=args['lambda_d'],  alpha=args['alpha'], gamma=gam, max_iter=iter, k_p=args['k_p'], K=args['K'], eta=args['eta'], pre=args['pre'])
            # if method=='mvfnmc1new1':
            if method== 'mvfnmc1new1':
                model = MvFNMC1new1(lambda_d=args['lambda_d'],  alpha=args['alpha'], gamma_d=gam1, gamma_t=gam2,max_iter=args['max_iter'], k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=args['delay'])
            if method== 'mvfnmc2':
                model = MvFNMC2(lam=args['lam'], lambda_d=args['lambda_d'],  alpha=args['alpha'], gamma_d=gam1, gamma_t=gam2, max_iter=args['max_iter'], k_p=args['k_p'],  K=args['K'], eta=args['eta'], pre=args['pre'], delay=args['delay'])

            
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            # print (cmd)
            aupr_vec, auc_vec,vec_drug, vec_tar = train(method,model, cv_data, dataset,Dm3,Tm3)
            #method,model, cv_data, dataset,Dm3,Tm3
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            cmd+="\n"+"auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic)
            results_list.append(cmd)
            
            if aupr_avg > max_aupr:
                max_aupr = aupr_avg
                aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
                # opt_para=[gam,iter]
                opt_para=[gam1,gam2]
    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    # cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print (cmd)
    results_list.append(cmd)
    out_file=os.path.join(output_dir, method+"_opt_gam"+str(cvs)+"_"+dataset+".txt")
    with open(out_file, "w", encoding="utf-8") as file:
    # 循环遍历字符串列表，将每个字符串写入文件
        for string in results_list:
            file.write(string + "\n")  # 每行结束后添加换行符

    return opt_para



# def svnmc_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
#     max_auc, auc_opt = 0, []
#     for r in [50, 100]:
#         for x in np.arange(-5, 2):
#             for y in np.arange(-5, 3):
#                 for z in np.arange(-5, 1):
#                     for t in np.arange(-3, 1):
#                         tic = time.clock()
#                         model = NRLMF(cfix=para['c'], K1=para['K1'], K2=para['K2'], num_factors=r, lambda_d=float(2)**(x), lambda_t=float(2)**(x), alpha=float(2)**(y), beta=float(2)**(z), theta=float(2)**(t), max_iter=100)
#                         cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
#                         print (cmd)
#                         aupr_vec, auc_vec = train(model, cv_data, X, D, T)
#                         aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
#                         auc_avg, auc_conf = mean_confidence_interval(auc_vec)
#                         print ("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
#                         if auc_avg > max_auc:
#                             max_auc = auc_avg
#                             auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
#     cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
#     cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
#     print (cmd)

def nrlmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for r in [50, 100]:
        for x in np.arange(-5, 2):
            for y in np.arange(-5, 3):
                for z in np.arange(-5, 1):
                    for t in np.arange(-3, 1):
                        tic = time.clock()
                        model = NRLMF(cfix=para['c'], K1=para['K1'], K2=para['K2'], num_factors=r, lambda_d=float(2)**(x), lambda_t=float(2)**(x), alpha=float(2)**(y), beta=float(2)**(z), theta=float(2)**(t), max_iter=100)
                        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                        print (cmd)
                        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                        print ("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
                        if auc_avg > max_auc:
                            max_auc = auc_avg
                            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print (cmd)


def netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(-6, 3):  # [-6, 2]
        for y in np.arange(-6, 3):  # [-6, 2]
            tic = time.clock()
            model = NetLapRLS(gamma_d=10**(x), gamma_t=10**(x), beta_d=10**(y), beta_t=10**(y))
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            print (cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print ("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print (cmd)


def blmnii_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(0, 1.1, 0.1):
        tic = time.clock()
        model = BLMNII(alpha=x, avg=False)
        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
        print (cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print ("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print (cmd)


def wnngip_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(0.1, 1.1, 0.1):
        for y in np.arange(0.0, 1.1, 0.1):
            tic = time.clock()
            model = WNNGIP(T=x, sigma=1, alpha=y)
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            print (cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print ("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print (cmd)


def kbmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for d in [50, 100]:
        tic = time.clock()
        model = KBMF(num_factors=d)
        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
        print (cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print ("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print (cmd)


def cmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_aupr, aupr_opt = 0, []
    for d in [50, 100]:
        for x in np.arange(-2, -1):
            for y in np.arange(-3, -2):
                for z in np.arange(-3, -2):
                    tic = time.clock()
                    model = CMF(K=d, lambda_l=2**(x), lambda_d=2**(y), lambda_t=2**(z), max_iter=30)
                    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                    print (cmd)
                    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                    print ("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
                    if aupr_avg > max_aupr:
                        max_aupr = aupr_avg
                        aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print (cmd)
