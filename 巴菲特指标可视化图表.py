import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import os
from matplotlib import dates as mdates

# 设置非交互式后端，支持GitHub Actions
plt.switch_backend('Agg')

def setup_chinese_font():
    """
    设置中文字体，优先使用本地 Hiragino Sans GB.ttc 字体文件。
    """
    # 首先尝试使用本地字体文件
    local_font_path = 'Hiragino Sans GB.ttc'
    if os.path.exists(local_font_path):
        print(f"成功找到本地中文字体: {local_font_path}")
        return fm.FontProperties(fname=local_font_path)
    
    # 设置字体优先级列表作为备选
    font_list = ['Hiragino Sans GB', 'PingFang SC', 'Noto Sans CJK SC', 'SimHei', 'Heiti TC', 'DejaVu Sans']
    
    # 通用字体设置
    plt.rcParams['font.sans-serif'] = font_list
    plt.rcParams['axes.unicode_minus'] = False
    
    # 尝试找到可用的中文字体
    for font_name in font_list:
        try:
            font_prop = fm.FontProperties(family=font_name)
            fm.findfont(font_prop)
            print(f"使用字体: {font_name}")
            return font_prop
        except Exception:
            continue
            
    print("警告: 未找到合适的中文字体，可能影响显示效果。")
    return None


def plot_buffett_indicator():
    """
    获取巴菲特指标数据，并进行可视化。
    - 阈值区域：根据通用阈值划分估值区域，提供直观参考。
    - 历史分位：量化当前估值在历史长河中的位置。
    """
    chinese_font_prop = setup_chinese_font()
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # 1. 获取数据
        print("正在获取巴菲特指标数据...")
        buffett_df = ak.stock_buffett_index_lg()
        print("数据获取成功！")
        
        # 2. 数据处理与计算
        buffett_df['日期'] = pd.to_datetime(buffett_df['日期'])
        buffett_df.set_index('日期', inplace=True)
        
        # 计算核心指标：总市值 / GDP
        buffett_df['巴菲特指标'] = buffett_df['总市值'] / buffett_df['GDP']
        
        # **优化**: 聚焦于2010年之后的数据
        buffett_df = buffett_df.loc['2010-01-01':]
        
        # 获取最新数据用于标题
        last_row = buffett_df.iloc[-1]
        current_indicator_value = last_row['巴菲特指标']
        total_percentile = last_row['总历史分位数']
        
        # 3. 绘制图表
        print("正在生成图表...")
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax1 = plt.subplots(figsize=(18, 9))
        
        # **人性化优化**: 绘制估值区域背景
        ax1.axhspan(0, 0.8, color='lightgreen', alpha=0.5, zorder=0, label='低估区域 (<0.8)')
        ax1.axhspan(0.8, 1.0, color='yellow', alpha=0.4, zorder=0, label='合理区域 (0.8-1.0)')
        ax1.axhspan(1.0, 1.2, color='orange', alpha=0.4, zorder=0, label='高估区域 (1.0-1.2)')
        ax1.axhspan(1.2, buffett_df['巴菲特指标'].max() + 0.1, color='lightcoral', alpha=0.5, zorder=0, label='危险区域 (>1.2)')

        # --- 绘制主坐标轴 (左侧Y轴) ---
        ax1.plot(buffett_df.index, buffett_df['巴菲特指标'], label='巴菲特指标', color='dodgerblue', linewidth=2, zorder=5)
        ax1.axhline(y=current_indicator_value, color='darkred', linestyle=':', linewidth=1.5, label=f'当前值: {current_indicator_value:.2%}')
        
        ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax1.tick_params(axis='y', labelcolor='dodgerblue')
        ax1.set_ylim(bottom=0) # 指标不会为负，从0开始

        # --- 绘制次坐标轴 (右侧Y轴) ---
        ax2 = ax1.twinx()
        ax2.plot(buffett_df.index, buffett_df['收盘价'], label='上证综指 (右轴)', color='purple', alpha=0.6, linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor='purple')

        # **X轴优化**: 设置主次刻度
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y年'))
        ax1.xaxis.set_minor_locator(mdates.MonthLocator())
        
        # --- 设置图表标题、标签和图例 ---
        title_text = f'巴菲特指标 (A股总市值/GDP) vs. 上证综指\n当前历史分位: {total_percentile:.2%}'
        if chinese_font_prop:
            ax1.set_title(title_text, fontproperties=chinese_font_prop, fontsize=22, pad=20)
            ax1.set_xlabel('日期', fontproperties=chinese_font_prop, fontsize=12)
            ax1.set_ylabel('总市值 / GDP', fontproperties=chinese_font_prop, fontsize=14, color='dodgerblue')
            ax2.set_ylabel('上证综指', fontproperties=chinese_font_prop, fontsize=14, color='purple')
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left', prop=chinese_font_prop)
        else:
            ax1.set_title(f'Buffett Indicator vs. Shanghai Composite\nCurrent Total Percentile: {total_percentile:.2%}', fontsize=22, pad=20)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Market Cap / GDP', fontsize=14, color='dodgerblue')
            ax2.set_ylabel('Shanghai Composite Index', fontsize=14, color='purple')
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        # **X轴优化**: 增加月度虚线网格
        ax1.grid(True, which='major', axis='x', linestyle='--', linewidth=0.7)
        ax1.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.4)
        ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)
        fig.tight_layout()
        
        print("图表生成完毕，正在保存...")
        plt.savefig('buffett_indicator.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("巴菲特指标图表已保存为 buffett_indicator.png")
        plt.close()

    except Exception as e:
        print(f"执行过程中发生错误: {e}")

if __name__ == '__main__':
    plot_buffett_indicator()
