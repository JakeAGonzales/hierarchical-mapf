import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("deep")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (15, 7),  # Adjusted for 3 plots
    'figure.dpi': 300,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'grid.linestyle': '--',
    'grid.alpha': 0.3
})

def create_plots(df):
    """Create publication-quality plots from the data."""
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)
    
    # 1. Success Rate Plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df['agents'], df['hcbs_success'], 'o-', label='HCBS', color='#1f77b4')
    ax1.plot(df['agents'], df['cbs_success'], 's-', label='CBS', color='#ff7f0e')
    ax1.set_xlabel('Number of Agents')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('(a) Algorithm Success Rate', pad=10)
    ax1.grid(True)
    ax1.legend()
    
    # 2. Computation Time Plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df['agents'], df['hcbs_time'], 'o-', label='HCBS', color='#1f77b4')
    mask = ~df['cbs_time'].isna()
    ax2.plot(df['agents'][mask], df['cbs_time'][mask], 's-', label='CBS', color='#ff7f0e')
    ax2.plot(df['agents'], df['routing_time'], '^-', label='Routing Game', color='#2ca02c')
    ax2.set_xlabel('Number of Agents')
    ax2.set_ylabel('Computation Time (seconds)')
    ax2.set_title('(b) Average Computation Time', pad=10)
    ax2.grid(True)
    ax2.legend()
    
    # Format y-axis for seconds
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    
    # 3. Path Cost Plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df['agents'], df['hcbs_cost'], 'o-', label='HCBS', color='#1f77b4')
    mask = ~df['cbs_cost'].isna()
    ax3.plot(df['agents'][mask], df['cbs_cost'][mask], 's-', label='CBS', color='#ff7f0e')
    ax3.set_xlabel('Number of Agents')
    ax3.set_ylabel('Average Path Cost')
    ax3.set_title('(c) Solution Quality', pad=10)
    ax3.grid(True)
    ax3.legend()
    
    # Adjust layout
    plt.tight_layout()
    return fig

def main():
    # Read the data with header names
    header = ['agents', 'hcbs_success', 'cbs_success', 'hcbs_time', 'hcbs_time_std', 
              'cbs_time', 'cbs_time_std', 'routing_time', 'routing_time_std', 
              'hcbs_cost', 'hcbs_cost_std', 'cbs_cost', 'cbs_cost_std']
    
    df = pd.read_csv('compare.txt', comment='#', names=header)
    
    # Create plots
    fig = create_plots(df)
    
    # Save the figure
    fig.savefig('mapf_comparison_results.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('mapf_comparison_results.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()