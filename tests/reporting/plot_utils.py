import matplotlib.pyplot as plt
import pandas as pd
import os

def save_table_as_image(data, title, save_path, column_widths=None):
    """
    Converts a dictionary or list of dicts to a high-resolution thesis table image.
    """
    plt.figure(figsize=(10, 6), dpi=300)
    plt.axis('off')
    
    # Process data into a dataframe if it's a dict
    if isinstance(data, dict):
        df = pd.DataFrame(list(data.items()), columns=['Metric', 'Value'])
    else:
        df = pd.DataFrame(data)
        
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colWidths=column_widths if column_widths else [0.4, 0.4]
    )
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    
    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f9f9f9')
    
    plt.title(title, fontsize=16, pad=20, weight='bold', fontfamily='serif')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', transparent=False)
    plt.close()
    print(f"[PLOT-UTIL] Results image saved to: {save_path}")

def save_scalability_plot(stats_dict, save_path):
    """
    Plots Latency vs Concurrent Users.
    """
    users = list(stats_dict.keys())
    latencies = [stats_dict[u]['avg_latency'] for u in users]
    throughput = [stats_dict[u]['throughput'] for u in users]
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    color = 'tab:blue'
    ax1.set_xlabel('Concurrent Users', fontweight='bold')
    ax1.set_ylabel('Avg Latency (ms)', color=color, fontweight='bold')
    ax1.plot(users, latencies, color=color, marker='o', linewidth=2, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Throughput (req/sec)', color=color, fontweight='bold')
    ax2.plot(users, throughput, color=color, marker='s', linestyle='--', linewidth=2, label='Throughput')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("System Scalability Audit: AtrionNet Backend", fontsize=14, weight='bold', pad=15)
    fig.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[PLOT-UTIL] Scalability plot saved to: {save_path}")
