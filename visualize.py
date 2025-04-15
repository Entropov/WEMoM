import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 模拟数据生成
timestamps = np.arange(0, 300, 1)  # 300秒数据
workload = np.clip(np.sin(timestamps*0.1) * 0.4 + 0.5 + np.random.normal(0,0.1,300), 0, 1)
class_probs = np.abs(np.random.dirichlet((1, 2, 3), 300) + np.random.normal(0,0.05,(300,3)))

# 修正子图类型定义
fig = make_subplots(
    rows=2, cols=2,
    specs=[
        [  # 第一行
            {"type": "xy", "rowspan": 2},  # 左列（主图）
            {"type": "xy"}                 # 右上方（柱状图）
        ],
        [  # 第二行
            None,                          # 左列已跨行
            {"type": "indicator"}          # 右下方（仪表）
        ]
    ],
    column_widths=[0.7, 0.3],
    row_heights=[0.7, 0.3],
    vertical_spacing=0.1,
    horizontal_spacing=0.15,
    subplot_titles=("实时工作负荷趋势", "置信度分布", "", "当前状态")
)

# 主工作负荷曲线（左列）
fig.add_trace(
    go.Scatter(
        x=timestamps,
        y=workload,
        mode='lines+markers',
        name='负荷指数',
        line=dict(color='#1f77b4', width=1.5),
        marker=dict(size=4, color='#1f77b4')
    ),
    row=1, col=1
)

# 置信度柱状图（右上方）- 现在使用xy坐标系
fig.add_trace(
    go.Bar(
        x=['低', '中', '高'],
        y=class_probs[-1],
        marker_color=['#2ca02c', '#ff7f0e', '#d62728'],
        width=0.6,
        text=np.round(class_probs[-1], 2),
        textposition='outside',
        showlegend=False
    ),
    row=1, col=2
)

# 配置柱状图坐标轴
fig.update_xaxes(title_text="负荷等级", row=1, col=2)
fig.update_yaxes(title_text="置信概率", range=[0,1], row=1, col=2)

# 负荷等级指示器（右下方）
current_class = np.argmax(class_probs[-1])
fig.add_trace(
    go.Indicator(
        mode="number+delta",
        value=workload[-1],
        number=dict(
            font=dict(size=28),
            valueformat=".2f"),
        delta=dict(
            reference=workload[-2],
            position="top"),
        gauge=dict(
            axis=dict(
                range=[0, 1],
                tickvals=[0.3, 0.7],
                ticktext=["低阈值", "高阈值"],
                tickcolor="black"),
            steps=[
                dict(range=[0,0.3], color="lightgreen"),
                dict(range=[0.3,0.7], color="moccasin"),
                dict(range=[0.7,1], color="lightcoral")]
        ),
        title=dict(text="当前负荷值", font=dict(size=16))
    ),
    row=2, col=2
)

fig.show()