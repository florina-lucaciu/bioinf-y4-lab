from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm

HANDLE = "florina-lucaciu"

EXPR_CSV = Path(f"data/work/{HANDLE}/lab07/expression_matrix.csv")
MODULES_CSV = Path(f"data/work/{HANDLE}/lab07/modules_{HANDLE}.csv")

PRECOMPUTED_ADJ_CSV: Optional[Path] = None

CORR_METHOD = "spearman"
USE_ABS_CORR = True
ADJ_THRESHOLD = 0.6
WEIGHTED = False

SEED = 42
TOPK_HUBS = 10
NODE_BASE_SIZE = 60
EDGE_ALPHA = 0.15

OUT_DIR = Path(f"labs/07_network_viz/submissions/{HANDLE}")
OUT_PNG = OUT_DIR / f"network_{HANDLE}.png"
OUT_HUBS = OUT_DIR / f"hubs_{HANDLE}.csv"

def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

def read_expression_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    print(f"Loaded expression matrix: {df.shape[0]} genes x {df.shape[1]} samples")
    return df

def read_modules_csv(path: Path) -> Dict[str, int]:
    df = pd.read_csv(path)
    if 'Gene' not in df.columns or 'Module' not in df.columns:
        raise ValueError("Modules CSV must have 'Gene' and 'Module' columns")
    
    gene2module = dict(zip(df['Gene'], df['Module']))
    print(f"Loaded {len(gene2module)} genes across {df['Module'].nunique()} modules")
    return gene2module

def correlation_to_adjacency(expr: pd.DataFrame,
                             method: str,
                             use_abs: bool,
                             threshold: float,
                             weighted: bool) -> pd.DataFrame:
    corr = expr.T.corr(method=method)
    
    if use_abs:
        corr = corr.abs()
    
    adj = corr.copy()
    adj[adj < threshold] = 0
    
    np.fill_diagonal(adj.values, 0)
    
    if not weighted:
        adj = (adj > 0).astype(int)
    
    n_edges = (adj > 0).sum().sum() // 2
    print(f"Adjacency matrix: {adj.shape[0]} nodes, ~{n_edges} edges (threshold={threshold})")
    
    return adj

def graph_from_adjacency(A: pd.DataFrame) -> nx.Graph:
    G = nx.from_pandas_adjacency(A)
    
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    
    print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if isolated:
        print(f"Removed {len(isolated)} isolated nodes")
    
    return G

def color_map_from_modules(nodes: Iterable[str], gene2module: Dict[str, int]) -> Dict[str, str]:
    modules = sorted(set(gene2module.values()))
    
    if len(modules) <= 10:
        colors = cm.tab10.colors
    elif len(modules) <= 20:
        colors = cm.tab20.colors
    else:
        colors = cm.tab20.colors
    
    module2color = {mod: colors[i % len(colors)] for i, mod in enumerate(modules)}
    
    node_colors = {}
    for node in nodes:
        mod = gene2module.get(node, -1)
        if mod == -1:
            node_colors[node] = '#CCCCCC'
        else:
            node_colors[node] = module2color[mod]
    
    return node_colors

def compute_hubs(G: nx.Graph, topk: int) -> pd.DataFrame:
    degrees = dict(G.degree())
    
    try:
        betweenness = nx.betweenness_centrality(G)
    except:
        betweenness = {node: 0 for node in G.nodes()}
    
    hub_data = []
    for node in G.nodes():
        hub_data.append({
            'Gene': node,
            'Degree': degrees[node],
            'Betweenness': betweenness[node]
        })
    
    df = pd.DataFrame(hub_data)
    df = df.sort_values('Degree', ascending=False)
    
    print(f"\nTop {topk} hub genes:")
    print(df.head(topk).to_string(index=False))
    
    return df.head(topk)

if __name__ == "__main__":
    print("=" * 60)
    print("Network Visualization & Hub Gene Analysis")
    print("=" * 60)
    
    ensure_exists(EXPR_CSV)
    ensure_exists(MODULES_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    expr = read_expression_matrix(EXPR_CSV)
    gene2module = read_modules_csv(MODULES_CSV)
    
    module_genes = set(gene2module.keys())
    expr_filtered = expr.loc[expr.index.intersection(module_genes)]
    print(f"\nFiltered to {expr_filtered.shape[0]} genes present in modules")
    
    if PRECOMPUTED_ADJ_CSV and Path(PRECOMPUTED_ADJ_CSV).exists():
        print(f"\nLoading precomputed adjacency from {PRECOMPUTED_ADJ_CSV}")
        A = pd.read_csv(PRECOMPUTED_ADJ_CSV, index_col=0)
        A = A.loc[A.index.intersection(module_genes), A.columns.intersection(module_genes)]
    else:
        print(f"\nComputing adjacency matrix (method={CORR_METHOD}, threshold={ADJ_THRESHOLD})")
        A = correlation_to_adjacency(
            expr_filtered,
            method=CORR_METHOD,
            use_abs=USE_ABS_CORR,
            threshold=ADJ_THRESHOLD,
            weighted=WEIGHTED
        )
    
    print("\nBuilding network graph...")
    G = graph_from_adjacency(A)
    
    print("\nAssigning node colors by module...")
    node_color_map = color_map_from_modules(G.nodes(), gene2module)
    node_colors = [node_color_map[node] for node in G.nodes()]
    
    print("\nIdentifying hub genes...")
    hubs_df = compute_hubs(G, TOPK_HUBS)
    hub_genes = set(hubs_df['Gene'])
    
    node_sizes = [NODE_BASE_SIZE * 5 if node in hub_genes else NODE_BASE_SIZE 
                  for node in G.nodes()]
    
    print("\nGenerating network visualization...")
    plt.figure(figsize=(14, 12))
    
    pos = nx.spring_layout(G, seed=SEED, k=0.5, iterations=50)
    
    nx.draw_networkx_edges(G, pos, alpha=EDGE_ALPHA, width=0.5, edge_color='gray')
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.8)
    
    hub_labels = {node: node for node in G.nodes() if node in hub_genes}
    nx.draw_networkx_labels(G, pos, labels=hub_labels, 
                            font_size=9, font_weight='bold')
    
    plt.title(f"Co-Expression Network\n{G.number_of_nodes()} genes, "
              f"{G.number_of_edges()} edges (threshold={ADJ_THRESHOLD})",
              fontsize=14, pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    print(f"\nSaving network visualization to {OUT_PNG}")
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saving hub genes to {OUT_HUBS}")
    hubs_df.to_csv(OUT_HUBS, index=False)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"  - Network figure: {OUT_PNG}")
    print(f"  - Hub genes: {OUT_HUBS}")
    print("=" * 60)