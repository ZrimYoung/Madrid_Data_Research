# Madrid Data Research | 马德里数据研究

This project involves analyzing various datasets related to traffic accidents in Madrid. The analysis uses Difference-in-Differences (DID) methods and conducts robustness checks using multiple time windows and parallel tests. The results are stored in Excel files, and Python scripts are used for data processing and statistical analysis.  
本项目涉及对与马德里交通事故相关的各种数据集进行分析。分析采用了DID方法，并通过多个时间窗口和平行检验进行稳健性检验。结果存储在 Excel 文件中，使用 Python 脚本进行数据处理和统计分析。

## Project Structure | 项目结构

The project is organized into the following folders:  
项目分为以下几个文件夹：

### 1. 单期DID (Single Period DID)
Contains data and results for a single period DID analysis.  
包含单期DID分析的数据和结果。
- **2018年**: Contains data for the year 2018 including traffic accidents, control variables, and model results.  
  包含2018年交通事故数据、控制变量和模型结果。
- **2022年**: Contains data for the year 2022 with similar data as for 2018.  
  包含2022年数据，与2018年类似。

### 2. 多期DID (Multiple Period DID)
Contains data and results for multiple periods DID analysis.  
包含多期DID分析的数据和结果。
- **CONTROL_ADD_PAND.xlsx**: Control variables for the multiple period DID.  
  多期DID的控制变量。
- **did_input_data_monthly.xlsx**: Monthly data for the DID analysis.  
  DID分析的月度数据。
- **DiD_model_results.xlsx**: Results of the multiple period DID model.  
  多期DID模型的结果。
- **did3.ipynb**: Jupyter Notebook for multiple period DID analysis.  
  多期DID分析的Jupyter Notebook。
- **TrafficAccident.xlsx**: Traffic accident data for multiple periods.  
  多期交通事故数据。

### 3. 平行检验 (Parallel Test)
This section includes tests to check the robustness of the results.  
此部分包含用于检查结果稳健性的检验。
- **parallel2018.png** & **parallel2022.png**: Graphs showing parallel trends for 2018 and 2022.  
  显示2018年和2022年平行趋势的图表。
- **parrale2022.ipynb & parrel2018.ipynb**: Jupyter Notebooks for parallel test analysis for both years.  
  2018年和2022年平行检验分析的Jupyter Notebook。

### 4. 短效性检验 (Short-term Effect Test)
Contains tests related to the short-term effect of the interventions on traffic accidents.  
包含与干预对交通事故短期影响相关的检验。
- **3M, 6M, 12M folders**: These folders contain data and results for short-term effects over 3, 6, and 12 months.  
  这些文件夹包含3个月、6个月和12个月的短期效应数据和结果。
- **short_long.py**: Python script for testing short- and long-term effects.  
  测试短期和长期效应的Python脚本。
