# import matplotlib.pyplot as plt

# # 原始数据
# datasets = ["Es", "IC", "GPCRs", "NR", "Luo"]
# n_list = [445, 210, 223, 54, 708]
# m_list = [664, 204, 95, 26, 1512]
# nm_list = [295480, 42840, 21185, 1404, 1070496]

# # MvFNMC1 数据
# train_time_c1 = [1208.887717, 395.056277, 348.419791, 0.141088, 6332.105032]
# aupr_c1       = [0.921001,    0.927759,    0.7749,     0.739852, 0.966]

# # MvFNMC2 数据
# train_time_c2 = [1205.771158, 436.703063, 396.969318, 0.166858, 5981.370519]
# aupr_c2       = [0.920746,    0.929853,    0.77759,    0.74588,  0.969]

# # 将各个列表打包在一起，方便排序
# zipped = list(zip(nm_list, datasets, n_list, m_list, train_time_c1, aupr_c1, train_time_c2, aupr_c2))

# # 根据 nm_list 排序，从小到大
# zipped_sorted = sorted(zipped, key=lambda x: x[0])

# # 解压排序后的数据
# nm_list_sorted, datasets_sorted, n_list_sorted, m_list_sorted, train_time_c1_sorted, aupr_c1_sorted, train_time_c2_sorted, aupr_c2_sorted = zip(*zipped_sorted)

# # 创建画布
# plt.figure(figsize=(12, 5))

# ###########################
# # 子图 1: Training Time vs. Dataset Size
# ###########################
# plt.subplot(1, 2, 1)
# plt.plot(nm_list_sorted, train_time_c1_sorted, 'o-', label="MvFNMC1", color='blue')
# plt.plot(nm_list_sorted, train_time_c2_sorted, 's-', label="MvFNMC2", color='orange')

# # 在数据点上标注 (n, m)
# for i in range(len(datasets_sorted)):
#     # plt.text(
#     #     nm_list_sorted[i], 
#     #     train_time_c1_sorted[i], 
#     #     f"({n_list_sorted[i]}, {m_list_sorted[i]})", 
#     #     fontsize=8, 
#     #     color='blue',
#     #     ha='center',
#     #     va='bottom'
#     # )
#     plt.text(
#         nm_list_sorted[i], 
#         train_time_c2_sorted[i], 
#         f"({n_list_sorted[i]}, {m_list_sorted[i]})", 
#         fontsize=8, 
#         color='orange',
#         ha='center',
#         va='top'
#     )

# plt.xlabel("Dataset Size (n × m)")
# plt.ylabel("Training Time (Sec)")
# plt.title("Training Time vs. Dataset Size")
# plt.legend()
# plt.grid(True)

# ###########################
# # 子图 2: AUPR vs. Dataset Size
# ###########################
# plt.subplot(1, 2, 2)
# plt.plot(nm_list_sorted, aupr_c1_sorted, 'o-', label="MvFNMC1", color='blue')
# plt.plot(nm_list_sorted, aupr_c2_sorted, 's-', label="MvFNMC2", color='orange')

# # 在数据点上标注 (n, m)
# for i in range(len(datasets_sorted)):
#     # plt.text(
#     #     nm_list_sorted[i], 
#     #     aupr_c1_sorted[i], 
#     #     f"({n_list_sorted[i]}, {m_list_sorted[i]})", 
#     #     fontsize=8, 
#     #     color='blue',
#     #     ha='center',
#     #     va='bottom'
#     # )
#     plt.text(
#         nm_list_sorted[i], 
#         aupr_c2_sorted[i], 
#         f"({n_list_sorted[i]}, {m_list_sorted[i]})", 
#         fontsize=8, 
#         color='orange',
#         ha='center',
#         va='top'
#     )

# plt.xlabel("Dataset Size (n × m)")
# plt.ylabel("AUPR")
# plt.title("AUPR vs. Dataset Size")
# plt.legend()
# plt.grid(True)

# # 调整布局并保存图片
# plt.tight_layout()
# plt.savefig("training_aupr_plot_sorted1.png", dpi=300, bbox_inches='tight')  # 保存图片
# plt.show()



import matplotlib.pyplot as plt

# 原始数据
datasets = ["Es", "IC", "GPCRs", "NR", "Luo"]
n_list = [445, 210, 223, 54, 708]
m_list = [664, 204, 95, 26, 1512]
nm_list = [295480, 42840, 21185, 1404, 1070496]

# MvFNMC1 数据
train_time_c1 = [1208.887717, 395.056277, 348.419791, 0.141088, 6332.105032]
aupr_c1       = [0.921001,    0.927759,    0.7749,     0.739852, 0.966]

# MvFNMC2 数据
train_time_c2 = [1205.771158, 436.703063, 396.969318, 0.166858, 5981.370519]
aupr_c2       = [0.920746,    0.929853,    0.77759,    0.74588,  0.969]

# 将各个列表打包在一起，方便排序
zipped = list(zip(nm_list, datasets, n_list, m_list, train_time_c1, aupr_c1, train_time_c2, aupr_c2))
# 根据 nm_list 排序，从小到大
zipped_sorted = sorted(zipped, key=lambda x: x[0])
# 解压排序后的数据
nm_list_sorted, datasets_sorted, n_list_sorted, m_list_sorted, train_time_c1_sorted, aupr_c1_sorted, train_time_c2_sorted, aupr_c2_sorted = zip(*zipped_sorted)

###############################################
# 图1: Training Time vs. Dataset Size 的绘制与保存
###############################################
plt.figure(figsize=(6, 5))
plt.plot(nm_list_sorted, train_time_c1_sorted, 'o-', label="MvFNMC1", color='blue')
plt.plot(nm_list_sorted, train_time_c2_sorted, 's-', label="MvFNMC2", color='orange')

# 在数据点标注 (n, m)
for i in range(len(datasets_sorted)):
    # plt.text(nm_list_sorted[i], train_time_c1_sorted[i],
    #          f"({n_list_sorted[i]}, {m_list_sorted[i]})",
    #          fontsize=8, color='blue', ha='center', va='bottom')
    plt.text(nm_list_sorted[i], train_time_c2_sorted[i],
             f"({n_list_sorted[i]}, {m_list_sorted[i]})",
             fontsize=8, color='orange', ha='center', va='top')

plt.xlabel("Dataset Size (n × m)")
plt.ylabel("Training Time (Sec)")
plt.title("Training Time vs. Dataset Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Training_Time_vs_Dataset_Size.png", dpi=300, bbox_inches='tight')
plt.show()

###############################################
# 图2: AUPR vs. Dataset Size 的绘制与保存
###############################################
plt.figure(figsize=(6, 5))
plt.plot(nm_list_sorted, aupr_c1_sorted, 'o-', label="MvFNMC1", color='blue')
plt.plot(nm_list_sorted, aupr_c2_sorted, 's-', label="MvFNMC2", color='orange')

# 在数据点标注 (n, m)
for i in range(len(datasets_sorted)):
    # plt.text(nm_list_sorted[i], aupr_c1_sorted[i],
    #          f"({n_list_sorted[i]}, {m_list_sorted[i]})",
    #          fontsize=8, color='blue', ha='center', va='bottom')
    plt.text(nm_list_sorted[i], aupr_c2_sorted[i],
             f"({n_list_sorted[i]}, {m_list_sorted[i]})",
             fontsize=8, color='orange', ha='center', va='top')

plt.xlabel("Dataset Size (n × m)")
plt.ylabel("AUPR")
plt.title("AUPR vs. Dataset Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("AUPR_vs_Dataset_Size.png", dpi=300, bbox_inches='tight')
plt.show()
