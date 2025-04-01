import os
import re
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

log_file = "process_log.txt"

def log_step(message):
    print(f"[INFO] {message}")
    with open(log_file, "a") as f:
        f.write(f"[INFO] {message}\n")

# 清空日志文件（在脚本开始时调用一次）
with open(log_file, "w") as f:
    f.write("日志开始记录:\n")

def load_traffic_data(file_path):
    log_step("加载交通事故数据...")
    df = pd.read_excel(file_path, sheet_name=None)
    traffic_df = pd.read_excel(file_path, sheet_name=0)
    log_step("成功读取交通事故数据")
    
    # 处理列名，转换 YYYYMM_rate 格式为日期格式
    traffic_df = traffic_df.melt(id_vars=["COD_DIS", "COD_BAR"], var_name="DATE", value_name="ACCIDENT_RATE")
    traffic_df["DATE"] = pd.to_datetime(traffic_df["DATE"].str.replace("_rate", ""), format="%Y%m")
    log_step("交通事故数据日期转换完成")
    
    return traffic_df

def load_social_factors(file_path, output_dir):
    log_step("加载社会因子数据...")
    xls = pd.ExcelFile(file_path)
    factor_dfs = []
    
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        df = df.melt(id_vars=["COD_DIS", "COD_BAR"], var_name="DATE", value_name=sheet.upper())
        df["DATE"] = pd.to_datetime(df["DATE"].str.extract(r'(\d{4}_\d{2})')[0], format="%Y_%m")
        factor_dfs.append(df)
    
    factors_df = factor_dfs[0]
    for other_df in factor_dfs[1:]:
        factors_df = factors_df.merge(other_df, on=["COD_DIS", "COD_BAR", "DATE"], how="outer")
    
    log_step("社会因子数据加载完成")
    """
    # 标准化处理
    scaler = StandardScaler()
    factor_columns = ["EDUCATION", "INCOME", "POPULATION", "TRANSPORT", "COMMERCIAL", "RESIDENTIAL", "ROADVOLUME"]
    factors_df[factor_columns] = scaler.fit_transform(factors_df[factor_columns])
    log_step("社会因子数据标准化完成")
    """
    # 归一化处理
    scaler = MinMaxScaler()
    factor_columns = ["EDUCATION", "INCOME", "POPULATION", "TRANSPORT", "COMMERCIAL", "RESIDENTIAL", "ROADVOLUME"]
    factors_df[factor_columns] = scaler.fit_transform(factors_df[factor_columns])
    log_step("社会因子数据归一化完成")
    """
    # 保存处理后的数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_output_path = os.path.join(output_dir, f"Normalized_Factors_{timestamp}.xlsx")
    factors_df.to_excel(norm_output_path, index=False)
    log_step(f"处理数据已保存至 {norm_output_path}")
    """
    return factors_df

def load_treat_policy(file_path):
    log_step("加载TREAT_POLICY数据...")
    df = pd.read_excel(file_path)
    log_step("TREAT_POLICY数据加载完成")
    return df
# 原有数据加载函数保持不变...

def run_did_analysis(traffic_path, factors_path, treat_path, output_dir):
    log_step("开始运行DiD分析...")
    
    # 加载并合并数据（保持原有逻辑不变）
    traffic_df = load_traffic_data(traffic_path)
    factors_df = load_social_factors(factors_path, output_dir)
    treat_df = load_treat_policy(treat_path)
    
    df = traffic_df.merge(factors_df, on=["COD_DIS", "COD_BAR", "DATE"], how="left")
    df = df.merge(treat_df, on=["COD_DIS", "COD_BAR"], how="left")
    log_step("数据合并完成")
    
    # 新增相对月份计算
    policy_date = pd.to_datetime('2018-12-01')
    df['rel_month'] = (df['DATE'].dt.to_period('M') - policy_date.to_period('M')).apply(lambda x: x.n)
    
    # 原有DID模型保持不变...
    
    # ========== 新增平行趋势检验 ==========
    log_step("开始平行趋势检验...")
    
    # 生成动态处理效应模型
    formula_pt = (
        "ACCIDENT_RATE ~ TREAT_POLICY * C(rel_month, Treatment(-1)) + "
        "EDUCATION + INCOME + POPULATION + TRANSPORT + COMMERCIAL + RESIDENTIAL + ROADVOLUME + "
        "C(COD_BAR, Treatment(1))"
    )
    
    model_pt = smf.ols(formula_pt, data=df).fit(
        cov_type='cluster', 
        cov_kwds={'groups': df['COD_BAR']}
    )
    
    # 提取系数结果
    results_pt = pd.DataFrame({
        'Variable': model_pt.params.index,
        'Coefficient': model_pt.params.values,
        'Std.Error': model_pt.bse.values,
        't-Statistic': model_pt.tvalues.values,
        'P-Value': model_pt.pvalues.values
    })
    
    # 提取政策前系数
    pre_coeffs = []
    pattern = re.compile(r'\[T\.(-?\d+)\]')
    for var in model_pt.params.index:
        if 'TREAT_POLICY:C(rel_month' in var:
            match = pattern.search(var)
            if match:
                m = int(match.group(1))
                if m < -1:  # 排除基准期-1
                    pre_coeffs.append(var)
    
    # 执行联合检验
    test_result = {}
    if pre_coeffs:
        joint_test = model_pt.wald_test(pre_coeffs)
        test_result = {
            'F-Statistic': joint_test.statistic,
            'P-Value': joint_test.pvalue,
            'Num_Coefficients': len(pre_coeffs)
        }
    else:
        test_result = {'Error': 'No pre-treatment coefficients found'}
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pt_path = os.path.join(output_dir, f"Parallel_Trends_Results_{timestamp}.xlsx")
    
    with pd.ExcelWriter(pt_path) as writer:
        results_pt.to_excel(writer, sheet_name='Dynamic_Coefficients', index=False)
        pd.DataFrame([test_result]).to_excel(writer, sheet_name='Joint_Test', index=False)
    
    log_step(f"平行趋势检验结果已保存至 {pt_path}")
    return model_pt.summary()

# 示例调用
run_did_analysis("TrafficAccident_18.xlsx", "CONTROL_ADD_18.xlsx", "MARK_18.xlsx", "did_results_18")
# 其余辅助函数保持不变...