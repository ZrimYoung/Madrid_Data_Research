import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# 日志函数，增加文件写入功能
log_file = "process_log.txt"

def log_step(message):
    print(f"[INFO] {message}")  # 控制台输出
    with open(log_file, "a") as f:  # 写入日志文件
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
    factor_columns = ["EDUCATION", "INCOME", "POPULATION", "TRANSPORT", "COMMERCIAL", "RESIDENTIAL" ]
    factors_df[factor_columns] = scaler.fit_transform(factors_df[factor_columns])
    log_step("社会因子数据标准化完成")
   """
    # 归一化处理
    scaler = MinMaxScaler()
    factor_columns = ["EDUCATION", "INCOME", "POPULATION", "TRANSPORT", "COMMERCIAL", "RESIDENTIAL", "PANDEMIC"]
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

def run_did_analysis(traffic_path, factors_path, treat_path, output_dir):
    log_step("开始运行DiD分析...")
    # 加载数据
    traffic_df = load_traffic_data(traffic_path)
    factors_df = load_social_factors(factors_path, output_dir)
    treat_df = load_treat_policy(treat_path)
    
    # 合并数据
    df = traffic_df.merge(factors_df, on=["COD_DIS", "COD_BAR", "DATE"], how="left")
    df = df.merge(treat_df, on=["COD_DIS", "COD_BAR"], how="left")
    log_step("数据合并完成")
    
    # 设定 POLICY（2018年12月之后为1，否则为0）
    df['POLICY'] = (df['DATE'] >= '2022-01-01').astype(int)
    log_step("POLICY 变量设定完成")
    
    # 确保 TREATMENT 列存在
    if 'TREAT_POLICY' not in df.columns:
        log_step("错误: 数据缺少 'TREAT_POLICY' 列")
        raise ValueError("数据缺少 'TREAT_POLICY' 列，请添加该列来标识受影响区域。")
    
    # 计算交互项 TREAT × POLICY
    df['TREAT_POLICY_INTERACT'] = df['TREAT_POLICY'] * df['POLICY']
    log_step("交互项计算完成")
    
    # 建立回归模型
    formula = "ACCIDENT_RATE ~ TREAT_POLICY + POLICY + TREAT_POLICY_INTERACT + EDUCATION + INCOME + POPULATION + TRANSPORT + COMMERCIAL + RESIDENTIAL + PANDEMIC + C(COD_BAR, Treatment(1)) + C(DATE, Treatment(1))"
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['COD_BAR']})
    log_step("回归模型拟合完成")
    
    # 计算模型统计指标
    model_stats = {
        "R-squared": model.rsquared,
        "Adj. R-squared": model.rsquared_adj,
        "F-statistic": model.fvalue,
        "Prob (F-statistic)": model.f_pvalue,
        "AIC": model.aic,
        "BIC": model.bic
    }
    log_step("模型统计指标计算完成")
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log_step("输出目录创建完成")
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"DID_results_{timestamp}.xlsx")
    
    # 保存结果到 Excel
    results_df = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values,
        'Std. Error': model.bse.values,
        't-Statistic': model.tvalues.values,
        'P-Value': model.pvalues.values
    })
    
    stats_df = pd.DataFrame(model_stats.items(), columns=['Statistic', 'Value'])
    
    with pd.ExcelWriter(output_path) as writer:
        results_df.to_excel(writer, sheet_name='DID Results', index=False)
        stats_df.to_excel(writer, sheet_name='Model Statistics', index=False)
    log_step(f"DiD 结果已保存至 {output_path}")
    return model.summary()

# 示例调用
run_did_analysis("TrafficAccident_22.xlsx", "CONTROL_ADD_PAND_22.xlsx", "MARK_22.xlsx", "did_results_22")
