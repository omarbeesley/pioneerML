
from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import torch
from typing import Optional, Tuple, Union, List, Sequence

from .utils import GraphRecord

def plot_event_display(
    records: Union[GraphRecord, Sequence[GraphRecord]],
    pred_points: Optional[Union[np.ndarray, torch.Tensor, List[List[float]]]] = None,
    pred_labels: Optional[Union[np.ndarray, List[int]]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    fixed_axes: bool = True,
    color_mode: str = 'pdg', # 'pdg', 'record', 'correctness', 'pred'
    title: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Generates an event display for one or more GraphRecord objects.
    
    Args:
        records: A single GraphRecord or a list/sequence of GraphRecords to overlay.
        pred_points: Optional predicted points [N, 3] or [N, 2, 3] etc. Flattened to list of points.
                     Each point is [x, y, z].
        pred_labels: Optional predicted per-hit labels [N_hits]. Should match truth PDG codes.
                     If provided, must match total number of hits in 'records'.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
        figsize: Size of the figure.
        fixed_axes: Whether to use fixed axis limits (legacy style).
        color_mode: Coloring strategy:
                    'pdg': Color by true particle type (hit_pdgs).
                    'record': Color by record/group ID.
                    'correctness': Green for correct prediction, Red for incorrect.
                    'pred': Color by predicted particle type (pred_labels).
        title: Optional custom title for the figure.

    Returns:
        The matplotlib Figure object.
    """
    
    # Normalize inputs
    if isinstance(records, GraphRecord):
        records_list = [records]
    else:
        records_list = list(records)

    # Process Predicted Points
    final_preds = []
    is_endpoint_pairs = False
    
    if pred_points is not None:
        if isinstance(pred_points, torch.Tensor):
            pts = pred_points.detach().cpu().numpy()
        elif isinstance(pred_points, list):
            pts = np.array(pred_points)
        else:
            pts = pred_points
            
        # Detect if we have [N, 2, 3] (pairs of endpoints)
        if pts.ndim == 3 and pts.shape[1] == 2 and pts.shape[2] == 3:
            final_preds = pts # Keep structure [N, 2, 3]
            is_endpoint_pairs = True
        else:
            # Flatten to (-1, 3) for generic points
            if pts.ndim > 1:
                pts = pts.reshape(-1, 3)
            final_preds = pts
            is_endpoint_pairs = False
    
    # Process Predicted Labels
    predictions = None
    if pred_labels is not None:
        if isinstance(pred_labels, torch.Tensor):
            predictions = pred_labels.detach().cpu().numpy().flatten()
        else:
            predictions = np.array(pred_labels).flatten()

    # Setup Figure
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Particle Colors Mapping
    particle_colors = {
        1: ("red", "Pion"),
        2: ("blue", "Muon"),
        3: ("purple", "Pion + Muon"),
        4: ("green", "MIP"),
        5: ("orange", "Pion + MIP"),
        6: ("cyan", "Muon + MIP"),
        0: ("gray", "Other")
    }
    default_color = ("gray", "Other")
    
    # Colormap for record mode (tab10 is good for distinct groups)
    cmap = plt.get_cmap('tab10')

    # Calculate global hit index offset for predictions
    global_hit_offset = 0

    # Plotting Loop (Views 0 and 1)
    for i in range(2):
        ax = axs[i]
        current_view_offset = 0
        
        # 1. Plot Hits for ALL records
        for idx, rec in enumerate(records_list):
            coord = np.array(rec.coord)
            z = np.array(rec.z)
            energy = np.array(rec.energy)
            view = np.array(rec.view)
            
            num_hits_in_rec = len(coord)
            
            if rec.hit_pdgs is not None:
                pdgs = np.array(rec.hit_pdgs).astype(int)
            else:
                pdgs = np.zeros_like(coord, dtype=int)
            
            # Extract predictions for this record if available
            rec_preds = None
            if predictions is not None:
                # Assuming predictions are concatenated in order of records
                # This logic requires careful alignment:
                # If we are in view loop i=0, we shouldn't consume offset.
                # Offset should be handled outside view loop or reset.
                # Actually, easier to slice:
                if i == 0: # Only calculate slice on first view pass
                     pass 
                # Wait, 'predictions' is for ALL hits (both views).
                # We need to slice predictions for this specific record.
                # BUT 'records_list' is iterated inside the view loop.
                # We need to compute the slice indices correctly.
                
                # To avoid re-computation complexity, let's look up by hit index.
                # We can trace global hit index.
                pass

            view_mask = (view == i)
            
            hZ = z[view_mask]
            hCoord = coord[view_mask]
            hE = energy[view_mask]
            hPDGs = pdgs[view_mask]
            
            # Get predictions corresponding to these specific hits
            # We need the INDICES of these hits in the original record
            # And the offset of this record in the global list
            
            # Let's simplify:
            # We need the global index for each hit to fetch its prediction.
            # Record 0 hits [0, N0), Record 1 hits [N0, N0+N1), etc.
            
            # This requires knowing start_index of 'rec' in the global predictions.
            # We can compute start_indices beforehand.
            pass

            if color_mode == 'record':
                # Use event_id if valid, else index
                e_id = rec.event_id
                c = cmap(idx % 10)
                colors = [c] * len(hZ)
                alphas = [0.6] * len(hZ)
            elif color_mode == 'classification':
                if predictions is None:
                    # Fallback if no preds
                    colors = ['gray'] * len(hZ)
                    alphas = [0.6] * len(hZ)
                else:
                    # Calculate start index for this record
                    start_idx = sum(len(r.coord) for r in records_list[:idx])
                    
                    # Indices of current view hits within the record
                    rec_indices = np.where(view_mask)[0]
                    
                    # Global indices
                    global_indices = start_idx + rec_indices
                    
                    rec_pred_slice = predictions[global_indices]
                    
                    # Determine alpha based on correctness
                    # Color based on TRUE PDG
                    colors = []
                    for p in hPDGs:
                        colors.append(particle_colors.get(p, default_color)[0])
                    
                    alphas = []
                    for true_pdg, pred_pdg in zip(hPDGs, rec_pred_slice):
                        if true_pdg == pred_pdg:
                            alphas.append(1.0) # Correct
                        else:
                            alphas.append(0.3) # Incorrect
                    
            elif color_mode == 'pred':
                if predictions is None:
                    colors = ['gray'] * len(hZ)
                else:
                    start_idx = sum(len(r.coord) for r in records_list[:idx])
                    rec_indices = np.where(view_mask)[0]
                    global_indices = start_idx + rec_indices
                    rec_pred_slice = predictions[global_indices]
                    
                    colors = [particle_colors.get(p, default_color)[0] for p in rec_pred_slice]
                alphas = [0.6] * len(hZ)
            else: # 'pdg' or unknown
                colors = [particle_colors.get(p, default_color)[0] for p in hPDGs]
                alphas = [0.6] * len(hZ)
            
            sizes = 100 * hE 
            
            # Scatter needs list of alphas or single alpha. Matplotlib scatter accepts alpha array?
            # No, standard matplotlib scatter alpha arg is scalar.
            # To have varying alpha, we must use RGBA colors.
            
            if color_mode == 'classification' and predictions is not None:
                # Convert colors to RGBA with per-point alpha
                rgba_colors = []
                for c, a in zip(colors, alphas):
                    # Get RGB from name
                    try:
                        rgb = mcolors.to_rgb(c)
                        rgba_colors.append(rgb + (a,))
                    except ValueError:
                        print(f"DEBUG: Invalid color {c}")
                        rgba_colors.append((0.5, 0.5, 0.5, a))
                        
                ax.scatter(hZ, hCoord, c=rgba_colors, s=sizes, edgecolors='none')
            else:
                # Use scalar alpha
                # If alphas list exists and is uniform, use first. Else...
                # For safety in other modes, use 0.6
                scalar_alpha = alphas[0] if alphas else 0.6
                ax.scatter(hZ, hCoord, c=colors, s=sizes, alpha=scalar_alpha, edgecolors='none')

            # 2. Plot True Endpoints (for this record)
            # Use same color as scatter if record mode, else black
            ep_color = cmap(idx % 10) if color_mode == 'record' else 'black'
            
            if rec.true_start is not None:
                ts = np.array(rec.true_start)
                ax.plot(ts[2], ts[i], marker='x', color=ep_color, markersize=10, markeredgewidth=2)
            if rec.true_end is not None:
                te = np.array(rec.true_end)
                ax.plot(te[2], te[i], marker='x', color=ep_color, markersize=12, markeredgewidth=2)

        # 3. Plot Predicted Points (if available) - Moved outside record loop for batch preds
        if len(final_preds) > 0:
            if is_endpoint_pairs:
                # final_preds shape is [N_records, 2, 3] or [1, 2, 3] broadcasted
                num_preds = len(final_preds)
                for p_idx, pair in enumerate(final_preds):
                    ps = pair[0]
                    pe = pair[1]
                    ax.plot(ps[2], ps[i], marker='+', color='red', markersize=12, markeredgewidth=2)
                    ax.plot(pe[2], pe[i], marker='+', color='red', markersize=12, markeredgewidth=2)
                    ax.plot([ps[2], pe[2]], [ps[i], pe[i]], color='red', linestyle='--', alpha=0.5)
            else:
                # Generic points [N, 3]
                z_preds = final_preds[:, 2]
                coord_preds = final_preds[:, i]
                ax.scatter(z_preds, coord_preds, marker='x', c='lime', s=100, zorder=10, label='Predicted' if i==0 else None)

        # 4. Styling
        ax.set_xlabel("z [mm]")
        ax.set_ylabel(f"{'x' if i == 0 else 'y'} [mm]")
        ax.set_title(f"{'x' if i == 0 else 'y'}-z View")
        
        if fixed_axes:
            ax.set_xlim(-0.2, 6.8)
            ax.set_ylim(-10.5, 10.5)
            ax.grid(True)
        else:
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.autoscale(enable=True, axis='both', tight=False)

        # Legend (View 0 only)
        if i == 0:
            legend_patches = []
            seen_labels = set()
            
            if color_mode == 'record':
                # Create legend for each record
                for idx in range(len(records_list)):
                    e_id = records_list[idx].event_id
                    label = f"Event {e_id}" if e_id is not None else f"Record {idx}"
                    c = cmap(idx % 10)
                    legend_patches.append(mpatches.Patch(color=c, label=label))
            else:
                # PDG or Pred or Classification Legend
                # In classification mode, we still show PDG Colors.
                if color_mode == 'pred' and predictions is not None:
                    source_pdgs = predictions
                else:
                    # Collect particle types from ALL records (True PDGs)
                    all_pdgs = []
                    for r in records_list:
                        if r.hit_pdgs is not None:
                            all_pdgs.extend(r.hit_pdgs)
                    source_pdgs = all_pdgs
                
                unique_pdgs = np.unique(source_pdgs) if len(source_pdgs) > 0 else []
                for p in unique_pdgs:
                    p_code = int(p)
                    c, l = particle_colors.get(p_code, (default_color, str(p_code)))
                    if l not in seen_labels:
                        legend_patches.append(mpatches.Patch(color=c, label=l))
                        seen_labels.add(l)
                

            
            # Add markers
            if color_mode == 'pdg' or color_mode == 'pred':
                legend_patches.append(Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=8, label='True Start/End'))
            
            if len(final_preds) > 0:
                 legend_patches.append(Line2D([0], [0], marker='x', color='lime', linestyle='None', markersize=10, label='Predicted Start/End' if is_endpoint_pairs else 'Predicted'))

            if len(legend_patches) > 0:
                ax.legend(handles=legend_patches, loc='upper right')


    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Event display saved to {save_path}")
        
    if show:
        plt.show()
        
    return fig

def plot_event_pileup(
    dataset: Sequence[GraphRecord],
    event_id: int,
    save_path: Optional[str] = None,
    show: bool = True
) -> Optional[plt.Figure]:
    """
    Finds all GraphRecords with the specified event_id and plots them together.
    
    Args:
        dataset: A sequence (list/dataset) of GraphRecord objects.
        event_id: The event ID to filter for.
        save_path: Optional path to save the plot.
        show: Whether to show the plot.
        
    Returns:
        The Figure object if records found, else None.
    """
    # Find matching records
    matching_records = [r for r in dataset if r.event_id == event_id]
    
    if not matching_records:
        print(f"No records found for event ID {event_id}")
        return None
        
    print(f"Found {len(matching_records)} records for event ID {event_id}")
    
    return plot_event_display(
        matching_records,
        color_mode='record',
        title=f"Pileup for Event {event_id} ({len(matching_records)} groups)",
        save_path=save_path,
        show=show
    )

def plot_pion_stop_events(
    records: Union[GraphRecord, Sequence[GraphRecord]],
    pred_points: Optional[Union[np.ndarray, torch.Tensor, List[List[float]]]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    fixed_axes: bool = True,
    title: str = "Pion Stop Visualization"
) -> Optional[plt.Figure]:
    """
    Dedicated plotter for Pion Stop events.
    Visualizes:
    - Hits (Colored by PDG)
    - True Pion Stop (Green Star)
    - Predicted Pion Stop (Red X)
    """
    if isinstance(records, GraphRecord):
        records = [records]
        
    # Standardize predictions to numpy [N, 3]
    final_preds = []
    if pred_points is not None:
        if isinstance(pred_points, torch.Tensor):
            final_preds = pred_points.detach().cpu().numpy()
        elif isinstance(pred_points, list):
            final_preds = np.array(pred_points)
        else:
            final_preds = pred_points
            
        if final_preds.ndim == 1:
            final_preds = final_preds.reshape(1, -1)
            
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    
    # Iterate views: 0=x-z, 1=y-z
    for i in range(2):
        # Setup Axes
        ax[i].set_xlabel('z [mm]')
        ax[i].set_ylabel(f"{'x' if i == 0 else 'y'} [mm]")
        ax[i].set_title(f"{'x' if i == 0 else 'y'}-z View")
        
        if fixed_axes:
            ax[i].set_xlim(-0.2, 6.8)
            ax[i].set_ylim(-10.5, 10.5)
            ax[i].grid(True)
        else:
            ax[i].grid(True, linestyle='--', alpha=0.5)
            ax[i].autoscale(enable=True, axis='both', tight=False)
        
        # Track present PDGs for Legend
        present_pdgs = set()
        
        # Plot Records
        for idx, rec in enumerate(records):
            r_z = np.array(list(rec.z))
            r_c = np.array(list(rec.coord))
            r_v = np.array(list(rec.view))
            
            r_e = np.array(list(rec.energy))
            
            # Determine Colors based on custom bitmask PDG
            # Reusing the standard mapping found in plot_event_display
            particle_colors = {
                1: ("red", "Pion"),
                2: ("blue", "Muon"),
                3: ("purple", "Pion + Muon"),
                4: ("green", "MIP"),
                5: ("orange", "Pion + MIP"),
                6: ("cyan", "Muon + MIP"),
                0: ("gray", "Other")
            }
            
            colors = np.array(['gray'] * len(r_z), dtype=object)
            # Filter for current view
            mask = (r_v == i)
            if rec.hit_pdgs is not None:
                pdgs = np.array(list(rec.hit_pdgs), dtype=int)
                # Only iterate if we have pdgs
                for k, pdg in enumerate(pdgs):
                    c, _ = particle_colors.get(pdg, ("gray", "Other"))
                    colors[k] = c
                
                # Update legend tracking
                view_pdgs = pdgs[mask]
                present_pdgs.update(view_pdgs)
                
            if np.any(mask):
                sc = ax[i].scatter(r_z[mask], r_c[mask], c=colors[mask], s=100*r_e[mask], alpha=0.6, label='Hits' if idx==0 else "")
                
                # Create legend handles manually if needed, but scatter doesn't auto-legend colors easily without unique labels per call
                # We can add dummy handles below for legend

            
            # Plot True Pion Stop
            if rec.true_pion_stop is not None:
                ts = np.array(rec.true_pion_stop)
                val = ts[0] if i==0 else ts[1]
                z_val = ts[2]
                
                ax[i].plot(z_val, val, marker='x', color='black', markersize=10, zorder=20, label='True Pion Stop' if idx==0 else "")
            else:
                if idx == 0 and i == 0:
                    print("DEBUG: rec.true_pion_stop is None for this record!")

        # Plot Predictions (Red X)
        if len(final_preds) > 0:
            for p_idx, pt in enumerate(final_preds):
                val = pt[0] if i==0 else pt[1]
                z_val = pt[2]
                
                ax[i].plot(z_val, val, marker='x', color='lime', markersize=10, 
                          markeredgewidth=2.5, linestyle='None', zorder=21, 
                          label='Pred Pion Stop' if p_idx==0 else "")
                
        # Legend (Dynamic)
        import matplotlib.patches as mpatches
        legend_patches = []
        
        # We need access to the dictionary here too
        # Add patches for types present in this view
        sorted_pdgs = sorted(list(present_pdgs))
        for pdg_code in sorted_pdgs:
            if pdg_code in particle_colors and pdg_code != 0:
                c, label = particle_colors[pdg_code]
                legend_patches.append(mpatches.Patch(color=c, label=label))
            elif pdg_code == 0:
                 # Ensure 'Other' is handled if present
                 c, label = particle_colors[0]
                 legend_patches.append(mpatches.Patch(color=c, label=label))

        # Add markers
        from matplotlib.lines import Line2D
        legend_patches.append(Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=15, label='True Pion Stop'))
        legend_patches.append(Line2D([0], [0], marker='x', color='lime', linestyle='None', markersize=12, label='Pred Pion Stop'))
        
        ax[i].legend(handles=legend_patches, loc='upper right')



    if title:
        plt.suptitle(title, fontsize=16)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    if show:
        plt.show()
        
    return fig
