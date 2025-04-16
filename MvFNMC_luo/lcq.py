import pandas as pd
from functions import get_drugs_targets_names

def find_high_score_false_negatives(n=20):
    # 文件路径
    datafile = '../luoout/output_AvgWei/mvfnmc2_predictions_cvs1_luo.csv'
    output_file = '../luoout/output_AvgWei/potential_new_links.csv'
    
    # 加载药物和靶标名称
    drug_names, target_names = get_drugs_targets_names('luo', 'luo')
    
    # 读取原始数据文件
    df = pd.read_csv(datafile)
    
    # 修正药物ID（因为名称列表少了一个元素）
    df['drug_id'] = df['drug_id'] - 1
    
    # 添加药物名称和靶标名称列
    df['drug_name'] = df['drug_id'].apply(lambda x: drug_names[x])
    df['target_name'] = df['target_id'].apply(lambda x: target_names[x])
    
    # 筛选标签为0但预测分数高的潜在新关联
    false_negatives = df[df['label'] == 0].copy()
    
    # 按预测分数降序排序
    false_negatives_sorted = false_negatives.sort_values(by='prediction', ascending=False)
    
    # 选择前N个结果
    top_potential = false_negatives_sorted.head(n)
    
    # 添加文献验证状态列（初始为空）
    top_potential['validated_in_literature'] = ''
    top_potential['reference'] = ''
    
    # 保存结果
    top_potential.to_csv(output_file, index=False, 
                        columns=['drug_name', 'drug_id', 'target_name', 'target_id', 
                                'prediction', 'label', 'validated_in_literature', 'reference'])
    
    # 打印结果
    print(f"\n发现{len(top_potential)}个潜在新关联（标签为0但预测分数高）:")
    print(top_potential[['drug_name', 'target_name', 'prediction']].to_string(index=False))
    
    print(f"\n完整结果已保存到: {output_file}")
    print("请查阅文献验证这些潜在关联，并在CSV文件的'validated_in_literature'和'reference'列添加验证结果")

if __name__ == "__main__":
    # 设置要输出的潜在关联数量
    NUM_POTENTIAL_LINKS = 50  # 可以根据需要调整
    find_high_score_false_negatives(NUM_POTENTIAL_LINKS)