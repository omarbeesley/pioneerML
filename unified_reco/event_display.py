import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import argparse

# --- CONSTANTS ---
PION      = 0b000001
MUON      = 0b000010
POSITRON  = 0b000100
ELECTRON  = 0b001000
GAMMA     = 0b010000
OTHER     = 0b100000

PARTICLE_COLORS = {
    PION: ("red", "Pion"),
    MUON: ("blue", "Muon"),
    POSITRON: ("green", "Positron"),
    ELECTRON: ("orange", "Electron"),
    GAMMA: ("cyan", "Gamma"),
    OTHER: ("gray", "Other")
}

def get_pdg_color(pdg_mask):
    # Retrieve highest priority particle type if multiple bits are set
    # Priority: ELECTRON (beam), POSITRON, MUON, PION, GAMMA, OTHER
    if pdg_mask & POSITRON: return PARTICLE_COLORS[POSITRON][0]
    if pdg_mask & ELECTRON: return PARTICLE_COLORS[ELECTRON][0]
    if pdg_mask & MUON: return PARTICLE_COLORS[MUON][0]
    if pdg_mask & PION: return PARTICLE_COLORS[PION][0]
    if pdg_mask & GAMMA: return PARTICLE_COLORS[GAMMA][0]
    return PARTICLE_COLORS[OTHER][0]

def plot_event(df, event_idx, color_by='pdg', save_path=None):
    row = df.iloc[event_idx]
    
    # ATAR Hit Extraction
    atar_x = np.array(row['atar_x'])
    atar_y = np.array(row['atar_y'])
    atar_z = np.array(row['atar_z'])
    atar_E = np.array(row['atar_E'])
    atar_view = np.array(row['atar_view'])  # 0 for XZ, 1 for YZ
    atar_pdg = np.array(row['atar_pdg'])
    
    try:
        atar_slice = np.array(row.get('atar_slice', np.zeros_like(atar_E)))
        atar_origin = np.array(row.get('atar_origin', np.zeros_like(atar_E)))
    except KeyError:
        atar_slice = np.zeros_like(atar_E)
        atar_origin = np.zeros_like(atar_E)

    try:
        lyso_E = np.array(row['lyso_E'])
        lyso_t = np.array(row['lyso_t'])
        lyso_pdg = np.array(row['lyso_pdg'])
        lyso_slice = np.array(row.get('lyso_slice', np.zeros_like(lyso_E)))
        # Calorimeter doesn't track discrete origins in native ROOT, but it DOES via the Pileup Mixer
        lyso_origin = np.array(row.get('lyso_origin', np.zeros_like(lyso_E))) 
    except KeyError:
        lyso_E, lyso_t, lyso_pdg, lyso_slice, lyso_origin = [], [], [], [], []

    fig, axs = plt.subplots(1, 3, figsize=(18, 12))
    fig.suptitle(f"Event Display: Index {event_idx} | Colored by {color_by.upper()}", fontsize=16)

    # CMaps for dynamic features
    cmap_tab10 = plt.get_cmap('tab10')
    # Colormaps for Slice/Origin Modes
    cmap_slice = plt.get_cmap('tab20')
    cmap_origin = plt.get_cmap('Set1')

    for i in range(2):
        ax = axs[i]

        # Plot ATAR
        atar_mask = (atar_view == i)
        hz = atar_z[atar_mask]
        hc = atar_x[atar_mask] if i == 0 else atar_y[atar_mask]
        he = atar_E[atar_mask]
        hslices = atar_slice[atar_mask]
        horigins = atar_origin[atar_mask]
        hpdgs = atar_pdg[atar_mask]

        sizes_atar = np.clip(he * 200, 10, 100)

        atar_colors = []
        for j in range(len(hz)):
            if color_by == 'pdg':
                atar_colors.append(get_pdg_color(hpdgs[j]))
            elif color_by == 'slice':
                atar_colors.append(cmap_slice(hslices[j] % 20))
            elif color_by == 'origin':
                atar_colors.append(cmap_origin(horigins[j] % 10))

        if len(hz) > 0:
            ax.scatter(hz, hc, c=atar_colors, s=sizes_atar, edgecolors='none', alpha=0.8)

        # Labels & Grids (Matching old script)
        ax.set_xlabel("z [mm]")
        ax.set_ylabel(f"{'x' if i == 0 else 'y'} [mm]")
        ax.set_title(f"{'x' if i == 0 else 'y'}-z View")
        
        # Fixed limits for ATAR
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-11.5, 11.5)
        ax.yaxis.set_major_locator(MultipleLocator(2.5))
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add Legend
        if i == 0:
            handles = []
            if color_by == 'pdg':
                for pdg, (col, label) in PARTICLE_COLORS.items():
                    handles.append(mpatches.Patch(color=col, label=label))
            elif color_by == 'slice':
                unique_slices = np.unique(atar_slice)
                for sid in unique_slices:
                    handles.append(mpatches.Patch(color=cmap_slice(sid % 20), label=f'Slice {sid}'))
            elif color_by == 'origin':
                unique_origins = np.unique(atar_origin)
                for oid in unique_origins:
                    handles.append(mpatches.Patch(color=cmap_origin(oid % 10), label=f'Origin {oid}'))
            
            ax.legend(handles=handles, loc='upper right')

    # Plot 3: Energy vs Time
    ax3 = axs[2]
    
    # ATAR Hit Arrays
    atar_E_all = np.array(row['atar_E'])
    atar_t_all = np.array(row['atar_t'])
    atar_pdg_all = np.array(row['atar_pdg'])
    atar_slice_all = np.array(row.get('atar_slice', np.zeros_like(atar_E_all)))
    atar_origin_all = np.array(row.get('atar_origin', np.zeros_like(atar_E_all)))

    atar_colors_all = []
    for j in range(len(atar_E_all)):
        if color_by == 'pdg':
            atar_colors_all.append(get_pdg_color(atar_pdg_all[j]))
        elif color_by == 'slice':
            atar_colors_all.append(cmap_slice(atar_slice_all[j] % 20))
        elif color_by == 'origin':
            atar_colors_all.append(cmap_origin(atar_origin_all[j] % 10))
            
    if len(atar_E_all) > 0:
        ax3.scatter(atar_t_all, atar_E_all, c=atar_colors_all, marker='o', s=30, alpha=0.8, label="ATAR")
        
    # LYSO Hit Arrays
    lyso_colors_all = []
    for j in range(len(lyso_E)):
        if color_by == 'pdg':
            lyso_colors_all.append(get_pdg_color(lyso_pdg[j]))
        elif color_by == 'slice':
            lyso_colors_all.append(cmap_slice(lyso_slice[j] % 20))
        elif color_by == 'origin':
            lyso_colors_all.append(cmap_origin(lyso_origin[j] % 10))
            
    if len(lyso_E) > 0:
        ax3.scatter(lyso_t, lyso_E, c=lyso_colors_all, marker='x', s=60, alpha=0.9, label="LYSO")
        ax3.set_yscale('log')
    ax3.set_xscale('symlog', linthresh=10.0)
    ax3.set_xlabel('Global Time (ns)')
    ax3.set_ylabel('Energy Deposit (Log Scale)')
    ax3.set_title('Global Energy vs Time (Matching Colors)')
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    if save_path and not save_path.endswith('.pdf'):
        plt.savefig(save_path, dpi=150)
        print(f"Saved event display to {save_path}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PIONEER PURITY Event Display")
    parser.add_argument("--input", type=str, required=True, help="Input Parquet file containing unrolled events.")
    parser.add_argument("--event", type=int, default=0, help="Event index to plot.")
    parser.add_argument("--num_events", type=int, default=1, help="Number of consecutive events to plot (for PDF).")
    parser.add_argument("--mode", type=str, choices=['pdg', 'slice', 'origin', 'all'], default='pdg', help="Color nodes by this mode.")
    parser.add_argument("--output", type=str, default="event_display.png", help="Path to save output image.")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    
    if args.event >= len(df):
        print(f"Error: Event {args.event} out of bounds. DataFrame has {len(df)} events.")
        exit(1)

    if args.output.endswith('.pdf') and args.num_events > 1:
        from matplotlib.backends.backend_pdf import PdfPages
        from tqdm import tqdm
        print(f"Generating PDF with {args.num_events} events...")
        with PdfPages(args.output) as pdf:
            end_idx = min(args.event + args.num_events, len(df))
            modes = ['pdg', 'slice', 'origin'] if args.mode == 'all' else [args.mode]
            for i in tqdm(range(args.event, end_idx)):
                for mode in modes:
                    fig = plot_event(df, i, color_by=mode, save_path=None)
                    pdf.savefig(fig)
                    plt.close(fig)
        print(f"Saved multi-event PDF to {args.output}")
    else:
        fig = plot_event(df, args.event, color_by=args.mode, save_path=args.output)
        if not args.output:
            plt.show()
