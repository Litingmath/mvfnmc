import numpy as np
# from scipy.stats import wilcoxon

# sample1 = np.array([2, 5, 2, 2, 3, 2, 1, 1.5, 2, 2.5, 3.5, 5])
# sample2 = np.array([1, 3, 1, 1, 2, 3, 2, 1.5, 1, 5, 3.5, 6])

# mvgnrlmf=np.array([5,4,9,9,8,7,6,10,5,6,9,10])
# dlmf=np.array([10,10,10,10,6,5.5,	8,	5,	9,	8,	6,	2,])
# mktcmf=np.array([4,	2,	4,	7,	4,	4,	4.5,	9,	4,	1,	7,	8])


# mvglp=np.array([3,7,7,8,1,1,3,8,3,2.5,1,9])

# srcmf=np.array([6,	1,	3,	4,	7,	8,	4.5,	3.5,	7.5,	9,	5,	3])
# grmf=np.array([8,	8,	8,	5,	5,	5.5,	7,	3.5,	6,	4,	2,	1])
# nrlmf=np.array([7,	9,	5,	3,	9,	9,	9,	6,	7.5,	7,	8,	4])
# cmf=np.array([9,	6,	6,	6,	10,	10,	10,	7,	10,	10,	10,	7])

# # 计算Wilcoxon符号秩检验
# # stat, p = wilcoxon(sample1, dlmf)
# # print('Statistics=%.3f, p=%.3f' % (stat, p))
# # stat, p = wilcoxon(sample2, dlmf)
# # print('Statistics=%.3f, p=%.3f' % (stat, p))
# # stat, p = wilcoxon(sample1, mktcmf)
# # print('Statistics=%.3f, p=%.3f' % (stat, p))
# # stat, p = wilcoxon(sample2, mktcmf)
# # print('Statistics=%.3f, p=%.3f' % (stat, p))
# # stat, p = wilcoxon( mvglp,sample1)
# # print('Statistics=%.3f, p=%.3f' % (stat, p))
# # stat, p = wilcoxon(mvglp,sample2)
# # print('Statistics=%.3f, p=%.3f' % (stat, p))

# stat, p = wilcoxon(sample1, srcmf)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# stat, p = wilcoxon(sample2, srcmf)
# print('Statistics=%.3f, p=%.3f' % (stat, p))

# stat, p = wilcoxon(sample1, grmf)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# stat, p = wilcoxon(sample2, grmf)
# print('Statistics=%.3f, p=%.3f' % (stat, p))

# stat, p = wilcoxon(sample1, nrlmf)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# stat, p = wilcoxon(sample2, nrlmf)
# print('Statistics=%.3f, p=%.3f' % (stat, p))

# stat, p = wilcoxon(sample1, cmf)
# print('Statistics=%.3f, p=%.3f' % (stat, p))
# stat, p = wilcoxon(sample2, cmf)
# print('Statistics=%.3f, p=%.3f' % (stat, p))

# #### bar plot
# import matplotlib.pyplot as plt

# drug_list=['L$_{1,d}$','L$_{2,d}$','L$_{3,d}$','L$_{4,d}$']
# tar_list=['L$_{1,t}$','L$_{2,t}$','L$_{3,t}$','L$_{4,t}$']
# # # Es
# # drug_w1=np.array([0.0577, 0.2649, 0.3740, 0.3034])
# # drug_w2=np.array([0.0227,0.2745, 0.3899, 0.3129])
# # tar_w1=np.array([ 0.3509, 0.2831, 0.0227, 0.3433])
# # tar_w2=np.array([0.3508,0.2837,0.0227,0.3428])
# # # # ic
# # drug_w1=np.array([0.0417, 0.2362, 0.4010,0.3211])
# # drug_w2=np.array([0.0227, 0.2476,0.4033,0.3264])
# # tar_w1=np.array([0.3631,  0.2334, 0.0227, 0.3807])
# # tar_w2=np.array([0.3567, 0.2618,0.0227, 0.3588])

# # # # gpcr
# # drug_w1=np.array([0.0833, 0.2357, 0.3928, 0.2881])
# # drug_w2=np.array([0.1250, 0.2436,0.3548, 0.2767])
# # tar_w1=np.array([0.3344, 0.2754,  0.0227, 0.3675])
# # tar_w2=np.array([0.3330,0.2880,0.0227,0.3563])

# # # nr
# drug_w1=np.array([0.1184, 0.2960,0.3373, 0.2483])
# drug_w2=np.array([0.1364, 0.2899,0.3272, 0.2466])
# tar_w1=np.array([0.4502, 0.0557, 0.0244, 0.4697])
# tar_w2=np.array([0.4515,0.1293,0.0227,0.3965])


# # x=list(range(len(drug_list)))
# x=np.arange(4)
# total_width, n=0.8, 2
# width=total_width/n

# # drug
# plt.figure(figsize=(5, 5))  # 创建一个图形窗口
# plt.bar(x-width/2,drug_w1,width, label='MvFNMC1',color='blue')
# # for i in range(len(x)):
# #     x[i]=x[i]+width
# plt.bar(x+width/2,drug_w2,width, label='MvFNMC2',color='orange')
# # plt.xlabel('Category')
# # plt.ylabel('Values')
# # plt.title('Grouped Bar Chart')
# plt.xticks(x, drug_list)
# plt.legend()
# plt.savefig('./weights/nr_drug.jpg')
# plt.show()
# plt.close()  # 关闭当前图形

# #tar
# plt.figure(figsize=(5, 5))  # 创建第二个图形窗口
# plt.bar(x-width/2,tar_w1,width, label='MvFNMC1',color='blue')
# # for i in range(len(x)):
# #     x[i]=x[i]+width
# plt.bar(x+width/2,tar_w2,width, label='MvFNMC2',color='orange')
# # plt.xlabel('Category')
# # plt.ylabel('Values')
# # plt.title('Grouped Bar Chart')
# plt.xticks(x, drug_list)
# plt.legend()
# plt.savefig('./weights/nr_tar.jpg')
# plt.show()
# plt.close()  # 关闭当前图形


#  # # 从文件读取数组
## iterations
# nr_aupr = np.load('m2_nr_re_aupr.npy')
# nr_auc  = np.load('m2_nr_re_auc.npy')
# gpcr_aupr = np.load('m2_gpcr_re_aupr.npy')
# gpcr_auc  = np.load('m2_gpcr_re_auc.npy')
# ic_aupr = np.load('m2_ic_re_aupr.npy')
# ic_auc  = np.load('m2_ic_re_auc.npy')
# e_aupr = np.load('m2_e_re_aupr.npy')
# e_auc  = np.load('m2_e_re_auc.npy')

## delay
nr_aupr = np.load('m1_nr_aupr_delay.npy')
gpcr_aupr = np.load('m1_gpcr_aupr_delay.npy')
ic_aupr = np.load('m1_ic_aupr_delay.npy')
e_aupr = np.load('m1_e_aupr_delay.npy')

# plot 
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))  # 创建一个图形窗口
x=np.arange(1, 11)
y1,y2,y3,y4=nr_aupr,gpcr_aupr,ic_aupr,e_aupr 
plt.plot(x, y1, marker='o', linestyle='-', color='blue', label='NR')
plt.plot(x, y2, marker='s', linestyle='--', color='green', label='GPCRs')
plt.plot(x, y3, marker='^', linestyle='-.', color='red', label='IC')
plt.plot(x, y4, marker='d', linestyle=':', color='purple', label='Es')

        
# 添加标题和标签
# plt.title("Line Chart with Multiple Lines", fontsize=16)
plt.xlabel("delay",fontsize=12)
plt.ylabel("AUPR",fontsize=12)
# plt.xticks(x, [1,3,5,7,9,11,])
# 添加网格
plt.grid(True, linestyle='--', alpha=0.5)
# 添加图例
plt.legend()

plt.savefig('./ablation/m1_aupr_delay.jpg')
# 显示图形
plt.show()
plt.close()  # 关闭当前图形     
        

# plt.figure(figsize=(6, 5))  # 创建一个图形窗口
# x=np.arange(1, 11)
# y1,y2,y3,y4=nr_auc,gpcr_auc,ic_auc,e_auc
# plt.plot(x, y1, marker='o', linestyle='-', color='blue', label='NR')
# plt.plot(x, y2, marker='s', linestyle='--', color='green', label='GPCRs')
# plt.plot(x, y3, marker='^', linestyle='-.', color='red', label='IC')
# plt.plot(x, y4, marker='d', linestyle=':', color='purple', label='Es')

        
# # 添加标题和标签
# # plt.title("Line Chart with Multiple Lines", fontsize=16)
# plt.xlabel("Iterations",fontsize=12)
# plt.ylabel("AUC",fontsize=12)
# # plt.xticks(x, [1,3,5,7,9,11,])
# # 添加网格
# plt.grid(True, linestyle='--', alpha=0.5)
# # 添加图例
# plt.legend()

# plt.savefig('./ablation/m2_auc_iter.jpg')
# # 显示图形
# plt.show()
# plt.close()  # 关闭当前图形     
        

