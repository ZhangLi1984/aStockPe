import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import os
import numpy as np
import logging
import json
import pytz
import google.generativeai as genai
from google.generativeai import types
import traceback
import akshare as ak

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 配置参数
# Use environment variable for API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DATA_DIR = "国际市场数据"  # 数据保存目录

# 创建数据目录
os.makedirs(DATA_DIR, exist_ok=True)

# 配置 Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logging.warning("警告: 未找到 GEMINI_API_KEY 环境变量。将无法生成 AI 分析报告。")

# 获取中国时间
def get_china_time():
    """获取中国时间"""
    china_tz = pytz.timezone('Asia/Shanghai')
    return datetime.now(china_tz)

# 格式化日期
def format_date(dt):
    """格式化日期为YYYY-MM-DD"""
    return dt.strftime('%Y-%m-%d')

# 读取本地巴菲特指标数据
def load_buffett_indicator_data():
    """读取本地保存的巴菲特指标数据"""
    try:
        if os.path.exists('buffett_indicator_data.csv'):
            logging.info("正在读取本地巴菲特指标数据...")
            buffett_df = pd.read_csv('buffett_indicator_data.csv', index_col=0, parse_dates=True, encoding='utf-8-sig')
            logging.info(f"成功读取巴菲特指标数据，共{len(buffett_df)}行")
            return buffett_df
        else:
            logging.warning("未找到本地巴菲特指标数据文件")
            return None
    except Exception as e:
        logging.error(f"读取巴菲特指标数据失败: {str(e)}")
        return None

# 读取本地股债利差数据
def load_equity_bond_spread_data():
    """读取本地保存的股债利差数据"""
    try:
        if os.path.exists('equity_bond_spread_data.csv'):
            logging.info("正在读取本地股债利差数据...")
            spread_df = pd.read_csv('equity_bond_spread_data.csv', index_col=0, parse_dates=True, encoding='utf-8-sig')
            logging.info(f"成功读取股债利差数据，共{len(spread_df)}行")
            return spread_df
        else:
            logging.warning("未找到本地股债利差数据文件")
            return None
    except Exception as e:
        logging.error(f"读取股债利差数据失败: {str(e)}")
        return None

# 获取中国国债收益率数据
def get_china_bond_yield():
    """获取中国10年期国债收益率数据"""
    try:
        logging.info("正在获取中国国债收益率数据...")
        # 使用AKShare获取中国国债收益率，指定起始日期为一年前
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        bond_data = ak.bond_zh_us_rate(start_date=start_date)
        
        # 打印获取到的数据结构
        print("获取到的国债数据结构:")
        print(bond_data.columns)
        print(bond_data.head())
        
        if bond_data.empty:
            logging.warning("未获取到国债收益率数据")
            return create_default_bond_data()
        
        # 检查是否有中国国债收益率10年列
        if '中国国债收益率10年' not in bond_data.columns:
            logging.warning("未找到'中国国债收益率10年'列，检查可用列")
            print("可用列:", bond_data.columns.tolist())
            return create_default_bond_data()
        
        # 创建新的DataFrame，只保留日期和中国国债收益率10年
        china_10y = pd.DataFrame({
            '日期': bond_data['日期'],
            '数值': bond_data['中国国债收益率10年']
        })
        
        # 删除NaN值
        china_10y = china_10y.dropna(subset=['数值'])
        
        if china_10y.empty:
            logging.warning("过滤NaN后数据为空")
            return create_default_bond_data()
            
        # 重命名列并设置索引
        china_10y.rename(columns={'日期': 'date', '数值': 'China_10Y_Treasury_Yield'}, inplace=True)
        china_10y['date'] = pd.to_datetime(china_10y['date'])
        china_10y.set_index('date', inplace=True)
        
        # 打印处理后的数据
        print("处理后的中国国债收益率数据:")
        print(china_10y.head())
        print(f"数据行数: {len(china_10y)}")
        
        logging.info("成功获取中国国债收益率数据")
        return china_10y
    except Exception as e:
        logging.error(f"获取中国国债收益率数据失败: {str(e)}")
        logging.error(traceback.format_exc())
        return create_default_bond_data()

def create_default_bond_data():
    """创建默认的中国国债收益率数据"""
    logging.info("创建默认的中国国债收益率数据")
    today = datetime.now()
    # 创建过去一年的每日数据
    dates = pd.date_range(end=today, periods=365, freq='D')
    china_10y = pd.DataFrame({
        'date': dates,
        'China_10Y_Treasury_Yield': [2.5] * len(dates)  # 使用2.5%作为默认值
    })
    china_10y.set_index('date', inplace=True)
    return china_10y

def download_gold_training_data(years=1, output_filename="gold_training_data_macro_enhanced.csv"):
    """
    Downloads a comprehensive financial dataset centered around gold, enhanced with key
    macroeconomic indicators, more commodities, and Chinese A-Share market data from Yahoo Finance.
    It then cleans the data, calculates percentage changes, computes the basis, and saves it to a CSV file.

    Args:
        years (int): The number of years of historical data to download.
        output_filename (str): The filename for the output CSV.
    """
    print("Executing financial data download script (Macro, Commodity & A-Shares Enhanced)...")

    # --- 1. Define Tickers to Download ---
    # This list uses ETFs as proxies for key macro indicators for more reliable data.
    tickers_to_download = {
        # --- Core Gold Assets ---
        'GC=F': 'GOLD_spot_price',         # 黄金现货价格 (通过近月期货代理)
        'MGC=F': 'GOLD_near_month_future', # 黄金近月期货 (微型合约，连续性好)

        # --- Other Precious & Industrial Metals ---
        'SI=F': 'SILVER_future',           # 白银期货
        'PL=F': 'PLATINUM_future',         # 铂金期货
        'HG=F': 'COPPER_future',           # 铜期货
        
        # --- NEW: Added More Industrial Metals & Materials (新增更多工业金属与原料) ---
        'ALI=F': 'ALUMINUM_future',        # 铝期货

        # --- Agricultural & Energy Commodity Futures ---
        'CL=F': 'OIL_price',               # 原油期货
        'NG=F': 'NATURAL_GAS_future',      # 天然气期货
        'ZC=F': 'CORN_future',             # 玉米期货
        'ZS=F': 'SOYBEANS_future',         # 大豆期货
        'ZW=F': 'WHEAT_future',            # 小麦期货
        'LE=F': 'LIVE_CATTLE_future',      # 活牛期货
        'HE=F': 'LEAN_HOGS_future',        # 瘦肉猪期货

        # --- Cryptocurrencies ---
        'BTC-USD': 'BTC_price',            # 比特币价格
        'ETH-USD': 'ETH_price',            # 以太币价格

        # --- US Market Sentiment & Volatility ---
        '^GSPC': 'SP500_close',            # 标普500指数
        '^IXIC': 'NASDAQ_close',           # 纳斯达克综合指数
        '^VIX': 'VIX_close',               # 波动率指数 (恐慌指数)

        # --- Chinese A-Share Market Indices ---
        '000001.SS': 'Shanghai_Composite_Index', # 上证综合指数
        '399001.SZ': 'Shenzhen_Component_Index', # 深证成份股指数
        '000300.SS': 'CSI_300_Index',            # 沪深300指数

        # --- Key Global Macro Indicators ---
        'DX-Y.NYB': 'US_Dollar_Index',         # 美元指数 (DXY)
        'CNY=X': 'USD_CNY_exchange_rate',  # 新增: 美元兑人民币汇率
        '^TNX': 'US_10Y_Treasury_Yield',   # 美国10年期国债收益率
        'TLT': 'Long_Term_Treasury_ETF',   # 20+年期美国国债ETF (代表长期利率和避险情绪)
        'HYG': 'High_Yield_Bond_ETF',      # 高收益公司债券ETF (代表市场风险偏好)
        'DBC': 'Commodity_Index_ETF',      # 综合商品指数ETF (代表通胀预期)
    }

    # --- 2. Set Date Range ---
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    print(f"Data download period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # --- 3. Execute Download ---
    print("\nDownloading data from Yahoo Finance, please wait...")
    try:
        full_data = yf.download(
            tickers=list(tickers_to_download.keys()),
            start=start_date,
            end=end_date,
            interval="1d",
            progress=True,
            threads=True
        )

        if full_data.empty:
            print("Error: No data was downloaded. Please check your network connection or ticker symbols.")
            return None

        # Use the 'Close' column for pricing data.
        data = full_data['Close']
        print("Data download successful!")

    except Exception as e:
        print(f"An error occurred during download: {e}")
        return None

    # --- 4. Data Cleaning and Processing ---
    print("\nPerforming data cleaning and processing...")
    main_df = data.rename(columns=tickers_to_download)
    main_df.sort_index(inplace=True)

    initial_cols = set(main_df.columns)
    main_df.dropna(axis=1, how='all', inplace=True)
    removed_cols = initial_cols - set(main_df.columns)
    if removed_cols:
        print(f"Warning: The following columns were removed for being completely empty: {list(removed_cols)}")

    main_df.ffill(inplace=True)
    main_df.bfill(inplace=True)
    print("Data cleaning complete.")

    # --- 5. Calculate Derivative Indicators ---
    print("Calculating gold near-month basis...")
    if 'GOLD_spot_price' in main_df.columns and 'GOLD_near_month_future' in main_df.columns:
        main_df['GOLD_basis_spot_vs_near'] = main_df['GOLD_spot_price'] - main_df['GOLD_near_month_future']
        print("Successfully calculated 'GOLD_basis_spot_vs_near' column.")
    else:
        print("Warning: Could not calculate basis, as Gold Spot or Near-Month Future data is missing.")
    
    # --- 6. Add China 10Y Treasury Yield data ---
    print("Adding China 10Y Treasury Yield data...")
    china_bond_data = get_china_bond_yield()
    if china_bond_data is not None and not china_bond_data.empty:
        # 确保索引格式一致
        main_df.index = pd.to_datetime(main_df.index)
        china_bond_data.index = pd.to_datetime(china_bond_data.index)
        
        print(f"主数据框索引范围: {main_df.index.min()} 到 {main_df.index.max()}")
        print(f"中国国债数据索引范围: {china_bond_data.index.min()} 到 {china_bond_data.index.max()}")
        
        # 确保中国国债数据的索引在主数据框的索引范围内
        filtered_china_data = china_bond_data[
            (china_bond_data.index >= main_df.index.min()) & 
            (china_bond_data.index <= main_df.index.max())
        ]
        
        if filtered_china_data.empty:
            print("警告: 过滤后的中国国债数据为空，尝试重采样数据")
            # 重采样中国国债数据以匹配主数据框的日期
            resampled_china_data = china_bond_data.resample('D').ffill()
            filtered_china_data = resampled_china_data[
                (resampled_china_data.index >= main_df.index.min()) & 
                (resampled_china_data.index <= main_df.index.max())
            ]
        
        # 合并数据
        if not filtered_china_data.empty:
            print(f"合并前主数据框形状: {main_df.shape}")
            
            # 使用reindex确保索引完全匹配
            aligned_china_data = filtered_china_data.reindex(main_df.index, method='ffill')
            main_df['China_10Y_Treasury_Yield'] = aligned_china_data['China_10Y_Treasury_Yield']
            
            print(f"合并后主数据框形状: {main_df.shape}")
            print(f"China_10Y_Treasury_Yield 列非空值数量: {main_df['China_10Y_Treasury_Yield'].count()}")
            print(f"China_10Y_Treasury_Yield 示例数据: {main_df['China_10Y_Treasury_Yield'].head()}")
        else:
            print("警告: 无法找到匹配的中国国债收益率数据，使用默认值")
            main_df['China_10Y_Treasury_Yield'] = 2.5  # 使用默认值
        
        print("Successfully added China 10Y Treasury Yield data.")
    else:
        print("Warning: Could not add China 10Y Treasury Yield data, using default value.")
        main_df['China_10Y_Treasury_Yield'] = 2.5  # 使用默认值

    # 填充可能的NaN值
    main_df.fillna(method='ffill', inplace=True)
    main_df.fillna(method='bfill', inplace=True)
    
    # 确保China_10Y_Treasury_Yield列存在
    if 'China_10Y_Treasury_Yield' not in main_df.columns:
        print("警告: 在最终数据中未找到China_10Y_Treasury_Yield列，创建默认值")
        main_df['China_10Y_Treasury_Yield'] = 2.5  # 使用默认值
    
    print("Final data processing complete!")

    # --- 7. Save to File ---
    try:
        output_path = os.path.join(os.getcwd(), output_filename)
        main_df.to_csv(output_path)
        print(f"\nSuccess! The integrated data has been saved to: {output_path}")
        print(f"Data dimensions (Rows, Columns): {main_df.shape}")
        print("Data preview (last 5 rows):")
        print(main_df.tail())
        
        return main_df

    except Exception as e:
        print(f"\nAn error occurred while saving the file: {e}")
        return None

def prepare_market_data_for_analysis(df, buffett_df=None, spread_df=None):
    """
    Prepares market data for AI analysis by extracting the most recent values.
    """
    if df is None or df.empty:
        return None
    
    latest_data = df.iloc[-1].to_dict()
    
    market_data = {
        "latest_date": df.index[-1].strftime('%Y-%m-%d'),
        "latest_values": {k: v for k, v in latest_data.items() if not k.endswith('_pct_change')}
    }
    
    # 添加巴菲特指标数据
    if buffett_df is not None and not buffett_df.empty:
        latest_buffett = buffett_df.iloc[-1]
        market_data["buffett_indicator"] = {
            "date": buffett_df.index[-1].strftime('%Y-%m-%d'),
            "indicator_value": float(latest_buffett['巴菲特指标']) if '巴菲特指标' in latest_buffett else None,
            "total_market_cap": float(latest_buffett['总市值']) if '总市值' in latest_buffett else None,
            "gdp": float(latest_buffett['GDP']) if 'GDP' in latest_buffett else None,
            "total_percentile": float(latest_buffett['总历史分位数']) if '总历史分位数' in latest_buffett else None,
            "close_price": float(latest_buffett['收盘价']) if '收盘价' in latest_buffett else None
        }
    
    # 添加股债利差数据
    if spread_df is not None and not spread_df.empty:
        latest_spread = spread_df.iloc[-1]
        market_data["equity_bond_spread"] = {
            "date": spread_df.index[-1].strftime('%Y-%m-%d'),
            "spread_value": float(latest_spread['股债利差']) if '股债利差' in latest_spread else None,
            "five_year_ma": float(latest_spread['5年均线']) if '5年均线' in latest_spread else None,
            "five_year_std": float(latest_spread['5年标准差']) if '5年标准差' in latest_spread else None,
            "plus_1_std": float(latest_spread['+1 STD']) if '+1 STD' in latest_spread else None,
            "minus_1_std": float(latest_spread['-1 STD']) if '-1 STD' in latest_spread else None,
            "plus_2_std": float(latest_spread['+2 STD']) if '+2 STD' in latest_spread else None,
            "minus_2_std": float(latest_spread['-2 STD']) if '-2 STD' in latest_spread else None,
            "csi300_index": float(latest_spread['沪深300指数']) if '沪深300指数' in latest_spread else None
        }
    
    return market_data

def generate_market_summary(market_data):
    """使用Gemini生成市场总结分析"""
    if not GEMINI_API_KEY:
        return "错误：未配置Gemini API KEY，无法生成市场分析。请设置 `GEMINI_API_KEY` 环境变量。"
    
    try:
        # Using a valid, current model name.
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
# 角色
你是一位专注于A股市场的顶级量化分析师，擅长通过巴菲特指标和股债利差等核心指标来判断市场风险与机会。

# 任务
基于巴菲特指标、股债利差以及全球市场数据，为A股投资者提供简洁明确的风险机会评估。
绿色区域 (<80%)：低估区域，历史上较好的买入时机
黄色区域 (80%-100%)：合理估值区域
橙色区域 (100%-120%)：高估区域，需要谨慎
红色区域 (>120%)：危险区域，历史上的泡沫水平

# 核心分析指标
## 1. 巴菲特指标分析
- **当前值**: {market_data.get('buffett_indicator', {}).get('indicator_value', 'N/A')}
- **历史分位**: {market_data.get('buffett_indicator', {}).get('total_percentile', 'N/A')}
- **估值判断**: 根据指标值判断当前A股整体估值水平（低估/合理/高估/危险）

## 2. 股债利差分析  
- **当前利差**: {market_data.get('equity_bond_spread', {}).get('spread_value', 'N/A')}
- **5年均线**: {market_data.get('equity_bond_spread', {}).get('five_year_ma', 'N/A')}
- **标准差位置**: 分析当前利差相对于±1σ、±2σ通道的位置
- **风险溢价判断**: 股票相对债券的吸引力如何

## 3. 市场环境
- **A股指数**: 上证综指、沪深300表现
- **全球环境**: VIX恐慌指数、美元指数、中美利差等宏观环境

# 输出要求
请按以下格式输出简洁的分析报告：
## 🎯 A股市场风险机会评估

### 📊 核心指标诊断
- **巴菲特指标**: [当前值] | [历史分位] | [估值结论]
- **股债利差**: [当前值] | [相对均线位置] | [风险溢价结论]

### ⚖️ 风险机会天平
**[机会大于风险 / 风险大于机会 / 风险机会均衡]**

### 🔍 关键逻辑
1. **估值逻辑**: 基于巴菲特指标的估值判断
2. **配置逻辑**: 基于股债利差的资产配置建议  
3. **环境逻辑**: 宏观环境对A股的影响
4. **技术分析**: K线形态、趋势线、支撑阻力位分析
5. **AI直觉**: 结合Gemini的模式识别能力判断市场情绪

### 💱 汇率影响分析
- **美元兑人民币汇率**: [当前汇率] | [短期变动方向] | [中期变动方向]
- **对沪深300影响**: 分析汇率短期和中期变动对沪深300指数的影响机制
- **传导路径**: 汇率变化方向通过资金流动、估值重构等路径影响A股
- **影响程度**: 评估汇率变动对沪深300未来走势的影响强度
### 📈 短期展望 (1-3个月)
- **市场方向**: 基于技术分析和AI直觉的综合判断
- **汇率驱动**: USD/CNY变化方向对A股短期走势的影响
- **关键技术位**: 重要支撑阻力位分析
- **风险提示**: 技术面和基本面的主要风险点

### 📊 技术分析要点
- **K线形态**: 分析最近的K线组合形态
- **趋势判断**: 短期、中期、长期趋势方向
- **量价关系**: 成交量与价格变化的配合情况
- **汇率技术**: USD/CNY技术形态对A股的指示意义
# 数据
```json
{json.dumps(market_data, ensure_ascii=False, indent=2)}
```

**要求**: 
- 总字数控制在600字以内
- 结论明确，避免模糊表述
- 重点突出风险与机会的对比
- 使用emoji增强可读性
"""

 
        
        logging.info("正在使用Gemini AI生成市场分析...")
        response = model.generate_content(
            prompt,
        )
        
        if response and hasattr(response, 'text'):
            logging.info("成功生成市场分析")
            return response.text
        else:
            logging.error("生成分析失败: 响应格式异常")
            logging.debug(f"Full API response: {response}")
            return "生成分析失败: 响应格式异常"
    
    except Exception as e:
        logging.error(f"生成分析失败: {str(e)}")
        logging.error(traceback.format_exc())
        return f"生成分析失败: {str(e)}"

def save_market_analysis_json(analysis_text):
    """保存市场分析为JSON格式，供网页展示使用"""
    try:
        today = get_china_time().strftime('%Y-%m-%d')
        
        # 保存为JSON格式供网页使用
        analysis_data = {
            "date": today,
            "analysis": analysis_text,
            "timestamp": get_china_time().isoformat()
        }
        
        json_filepath = "market_analysis.json"
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"市场分析已保存为JSON: {json_filepath}")
        return json_filepath
    except Exception as e:
        logging.error(f"保存市场分析JSON失败: {str(e)}")
        return None

def save_market_analysis(analysis_text, df):
    """保存市场分析到MD文件，并附上最近一个月的数据（按时间降序）。"""
    try:
        today = get_china_time().strftime('%Y-%m-%d')
        filename = f"market_analysis_{today}.md"
        filepath = os.path.join(DATA_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # 写入AI生成的分析
            f.write(analysis_text)
            f.write("\n\n---\n\n")
            f.write("## 最近一个月全部数据 (按时间降序排列)\n\n")

            # 截取最近一个月的数据
            if not df.empty:
                # 检查数据框中的列
                print(f"数据框中的所有列: {df.columns.tolist()}")
                print(f"China_10Y_Treasury_Yield 是否存在: {'China_10Y_Treasury_Yield' in df.columns}")
                
                last_date = df.index[-1]
                one_month_ago = last_date - pd.DateOffset(months=1)
                # 使用 .copy() 来避免 SettingWithCopyWarning
                last_month_df = df[df.index >= one_month_ago].copy()

                # 按用户要求，将数据按日期降序排列
                last_month_df.sort_index(ascending=False, inplace=True)

                # 所有列都是要显示的列
                display_columns = df.columns.tolist()
                print(f"将要显示的列: {display_columns}")
                
                # 确保重要列排在前面
                priority_columns = ['China_10Y_Treasury_Yield', 'US_10Y_Treasury_Yield', 'Shanghai_Composite_Index', 
                                   'CSI_300_Index', 'Shenzhen_Component_Index', 'GOLD_spot_price', 'OIL_price']
                
                # 重新排序列，优先显示重要列
                ordered_columns = []
                for col in priority_columns:
                    if col in display_columns:
                        ordered_columns.append(col)
                        display_columns.remove(col)
                
                # 添加剩余的列
                ordered_columns.extend(display_columns)
                print(f"最终排序后的列: {ordered_columns}")
                
                # 将数据转换为Markdown格式并写入文件
                if not last_month_df.empty and ordered_columns:
                    # 检查China_10Y_Treasury_Yield是否在最终列表中
                    if 'China_10Y_Treasury_Yield' not in ordered_columns and 'China_10Y_Treasury_Yield' in last_month_df.columns:
                        print("警告: China_10Y_Treasury_Yield不在最终列表中，但存在于数据框中")
                        ordered_columns.insert(0, 'China_10Y_Treasury_Yield')
                    
                    # 将索引（日期）格式化为字符串，避免时区信息
                    last_month_df.index = last_month_df.index.strftime('%Y-%m-%d')
                    
                    # 确保所有列都在数据框中
                    valid_columns = [col for col in ordered_columns if col in last_month_df.columns]
                    print(f"最终有效的列: {valid_columns}")
                    
                    # 转换为Markdown
                    markdown_table = last_month_df[valid_columns].to_markdown()
                    f.write(markdown_table)
        
        logging.info(f"市场分析及数据已保存到: {filepath}")
        return filepath
    except Exception as e:
        logging.error(f"保存市场分析及数据失败: {str(e)}")
        logging.error(traceback.format_exc())
        return None

def run_market_analysis():
    """运行完整的市场数据分析流程"""
    logging.info("开始下载最近1年的市场数据...")
    df = download_gold_training_data(years=1)
    
    if df is None:
        logging.error("数据下载失败，无法继续分析")
        return
    
    # 读取本地巴菲特指标和股债利差数据
    logging.info("读取本地巴菲特指标和股债利差数据...")
    buffett_df = load_buffett_indicator_data()
    spread_df = load_equity_bond_spread_data()
    
    logging.info("准备数据用于AI分析...")
    market_data = prepare_market_data_for_analysis(df, buffett_df, spread_df)
    
    if market_data is None:
        logging.error("数据准备失败，无法继续分析")
        return
    
    try:
        today = get_china_time().strftime('%Y-%m-%d')
        json_filename = f"global_market_data_{today}.json"
        json_filepath = os.path.join(DATA_DIR, "global_market_data_latest.json")
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(market_data, f, ensure_ascii=False, indent=2)
        
        dated_json_filepath = os.path.join(DATA_DIR, json_filename)
        with open(dated_json_filepath, 'w', encoding='utf-8') as f:
            json.dump(market_data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"市场数据已保存到: {json_filepath} 和 {dated_json_filepath}")
    except Exception as e:
        logging.error(f"保存市场数据JSON失败: {str(e)}")
    
    logging.info("使用Gemini AI生成市场分析...")
    analysis = generate_market_summary(market_data)
    
    if analysis and "生成分析失败" not in analysis:
        # 保存为JSON供网页使用
        json_path = save_market_analysis_json(analysis)
        
        # 保存完整的MD报告
        saved_path = save_market_analysis(analysis, df)
        
        if json_path and saved_path:
            logging.info(f"完整分析流程已完成")
            logging.info(f"网页JSON保存在: {json_path}")
            logging.info(f"详细报告保存在: {saved_path}")
        else:
            logging.error("保存分析报告失败")
    else:
        logging.error("生成分析失败，无法保存报告")

if __name__ == '__main__':
    run_market_analysis()
