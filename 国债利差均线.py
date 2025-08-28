import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import os
from matplotlib import dates as mdates

def setup_macos_chinese_font():
    """
    专门为 macOS 系统查找并设置可用的中文字体。
    会优先在系统字体目录中寻找 'PingFang.ttc'。
    """
    font_dirs = ['/System/Library/Fonts', '/Library/Fonts', '~/Library/Fonts']
    target_font = 'PingFang.ttc'
    
    for dir_path in font_dirs:
        font_path = os.path.join(os.path.expanduser(dir_path), target_font)
        if os.path.exists(font_path):
            print(f"成功找到 macOS 中文字体: {font_path}")
            return fm.FontProperties(fname=font_path)
            
    print("警告: 在 macOS 系统目录中未找到 'PingFang.ttc' 字体。")
    try:
        font_prop = fm.FontProperties(family='Heiti TC')
        fm.findfont(font_prop)
        print("备用方案：找到 'Heiti TC' 字体。")
        return font_prop
    except Exception:
        print("备用方案 'Heiti TC' 也未找到。")
    return None


def plot_equity_bond_spread():
    """
    获取股债利差数据，进行量化分析与可视化。
    - 5年周期：大致覆盖一个中短期经济周期，用于判断估值位置。
    - 标准差通道：识别极端估值区域。
    """
    chinese_font_prop = setup_macos_chinese_font()
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # 1. 获取数据
        print("正在获取股债利差数据...")
        stock_ebs_lg_df = ak.stock_ebs_lg()
        print("数据获取成功！")
        
        # 2. 数据处理
        stock_ebs_lg_df['日期'] = pd.to_datetime(stock_ebs_lg_df['日期'])
        stock_ebs_lg_df.set_index('日期', inplace=True)
        
        # **优化**: 聚焦于2010年之后的数据，使图表更清晰
        stock_ebs_lg_df = stock_ebs_lg_df.loc['2010-01-01':]
        
        # 定义5年滚动窗口 (5 * 252 ≈ 1260个交易日)
        window = 5 * 252
        
        # 3. 计算指标
        print("正在计算技术指标...")
        stock_ebs_lg_df['5年均线'] = stock_ebs_lg_df['股债利差'].rolling(window=window, min_periods=1).mean()
        stock_ebs_lg_df['5年标准差'] = stock_ebs_lg_df['股债利差'].rolling(window=window, min_periods=1).std()
        stock_ebs_lg_df['+1 STD'] = stock_ebs_lg_df['5年均线'] + 1 * stock_ebs_lg_df['5年标准差']
        stock_ebs_lg_df['-1 STD'] = stock_ebs_lg_df['5年均线'] - 1 * stock_ebs_lg_df['5年标准差']
        stock_ebs_lg_df['+2 STD'] = stock_ebs_lg_df['5年均线'] + 2 * stock_ebs_lg_df['5年标准差']
        stock_ebs_lg_df['-2 STD'] = stock_ebs_lg_df['5年均线'] - 2 * stock_ebs_lg_df['5年标准差']
        print("指标计算完成。")
        
        # 4. 绘制图表
        print("正在生成图表...")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax1 = plt.subplots(figsize=(18, 9))
        
        # --- 绘制主坐标轴 (左侧Y轴) ---
        ax1.plot(stock_ebs_lg_df.index, stock_ebs_lg_df['股债利差'], label='股债利差', color='royalblue', linewidth=1.5, zorder=5)
        ax1.plot(stock_ebs_lg_df.index, stock_ebs_lg_df['5年均线'], label='5年均线', color='orange', linestyle='--', linewidth=2, zorder=5)
        
        # 绘制标准差通道
        ax1.plot(stock_ebs_lg_df.index, stock_ebs_lg_df['+1 STD'], color='gray', alpha=0.6, linestyle=':', linewidth=1, label='±1 标准差')
        ax1.plot(stock_ebs_lg_df.index, stock_ebs_lg_df['-1 STD'], color='gray', alpha=0.6, linestyle=':', linewidth=1)
        ax1.plot(stock_ebs_lg_df.index, stock_ebs_lg_df['+2 STD'], color='gray', alpha=0.8, linestyle='--', linewidth=1.2, label='±2 标准差')
        ax1.plot(stock_ebs_lg_df.index, stock_ebs_lg_df['-2 STD'], color='gray', alpha=0.8, linestyle='--', linewidth=1.2)
        
        last_value = stock_ebs_lg_df['股债利差'].iloc[-1]
        ax1.axhline(y=last_value, color='darkred', linestyle=':', linewidth=1.5, label=f'当前值: {last_value:.2%}')

        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax1.tick_params(axis='y', labelcolor='royalblue')
        
        # --- 绘制次坐标轴 (右侧Y轴) ---
        ax2 = ax1.twinx()
        ax2.plot(stock_ebs_lg_df.index, stock_ebs_lg_df['沪深300指数'], label='沪深300指数 (右轴)', color='red', alpha=0.6, linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor='red')

        # **X轴优化**: 设置主次刻度
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y年'))
        ax1.xaxis.set_minor_locator(mdates.MonthLocator())
        
        # --- 设置图表标题、标签和图例 ---
        title_text = '股债利差 vs. 沪深300指数'
        if chinese_font_prop:
            ax1.set_title(title_text, fontproperties=chinese_font_prop, fontsize=22, pad=20)
            ax1.set_xlabel('日期', fontproperties=chinese_font_prop, fontsize=12)
            ax1.set_ylabel('股债利差 (风险溢价)', fontproperties=chinese_font_prop, fontsize=14, color='royalblue')
            ax2.set_ylabel('沪深300指数', fontproperties=chinese_font_prop, fontsize=14, color='red')
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left', prop=chinese_font_prop)
        else:
            ax1.set_title('Equity Bond Spread vs. CSI 300 Index', fontsize=22, pad=20)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Equity Bond Spread', fontsize=14, color='royalblue')
            ax2.set_ylabel('CSI 300 Index', fontsize=14, color='red')
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        # **X轴优化**: 增加月度虚线网格
        ax1.grid(True, which='major', axis='x', linestyle='--', linewidth=0.7)
        ax1.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.4)
        ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)
        fig.tight_layout()
        
        print("图表生成完毕，即将显示。")
        plt.show()

    except Exception as e:
        print(f"执行过程中发生错误: {e}")

if __name__ == '__main__':
    plot_equity_bond_spread()
