"""
Lineage Graph Visualizer
=========================
Creates interactive visualizations of lineage dependency graphs

Supports:
- Interactive network diagrams (Plotly)
- Static DAG visualizations (Graphviz)
- Hierarchical layouts
- Node filtering and highlighting

Author: STAG
Date: November 11, 2025
"""

from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import networkx as nx
from loguru import logger

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logger.warning("Graphviz not available - static graph export disabled")


class LineageVisualizer:
    """Visualize lineage dependency graphs"""

    @staticmethod
    def create_interactive_graph(
        lineage_graph: Dict[str, Any],
        highlight_path: Optional[List[str]] = None,
        layout: str = "hierarchical"
    ) -> go.Figure:
        """
        Create interactive Plotly graph visualization

        Args:
            lineage_graph: Lineage graph dict (from LineageGraph.to_dict())
            highlight_path: List of node IDs to highlight
            layout: Layout algorithm ("hierarchical", "spring", "circular")

        Returns:
            Plotly Figure object
        """
        nodes = lineage_graph.get('nodes', [])
        edges = lineage_graph.get('edges', [])

        if not nodes:
            logger.warning("No nodes in lineage graph")
            return LineageVisualizer._create_empty_graph()

        # Build NetworkX graph for layout calculation
        G = nx.DiGraph()

        for node in nodes:
            G.add_node(node['id'], **node)

        for edge in edges:
            G.add_edge(
                edge['source'],
                edge['target'],
                edge_type=edge.get('type', ''),
                script=edge.get('script', '')
            )

        # Calculate layout
        if layout == "hierarchical":
            pos = LineageVisualizer._hierarchical_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Create edge traces
        edge_traces = []

        for edge in edges:
            source_id = edge['source']
            target_id = edge['target']

            if source_id not in pos or target_id not in pos:
                continue

            x0, y0 = pos[source_id]
            x1, y1 = pos[target_id]

            # Determine edge color based on type
            edge_type = edge.get('type', '')
            if edge_type == 'reads':
                color = 'rgba(100, 149, 237, 0.5)'  # Blue
            elif edge_type == 'writes':
                color = 'rgba(60, 179, 113, 0.5)'  # Green
            else:
                color = 'rgba(128, 128, 128, 0.3)'  # Gray

            # Create arrow
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color=color),
                hoverinfo='text',
                text=f"{edge_type}: {edge.get('script', 'N/A')}",
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        node_symbols = []

        for node in nodes:
            node_id = node['id']
            if node_id not in pos:
                continue

            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)

            node_type = node.get('type', 'unknown')
            node_name = node.get('name', node_id)

            # Determine node appearance based on type
            if node_type == 'data_location':
                details = node.get('details', {})
                loc_type = details.get('location_type', 'unknown')

                if loc_type == 'hdfs_path':
                    color = '#FFD700'  # Gold
                    symbol = 'square'
                    size = 20
                elif loc_type == 'hive_table':
                    color = '#4169E1'  # Royal blue
                    symbol = 'diamond'
                    size = 25
                else:
                    color = '#87CEEB'  # Sky blue
                    symbol = 'circle'
                    size = 20

                hover_text = f"<b>{node_name}</b><br>Type: {loc_type}<br>Location: {node_id}"

            elif node_type == 'script':
                details = node.get('details', {})
                script_type = details.get('type', 'unknown')

                if script_type == 'source':
                    color = '#90EE90'  # Light green
                elif script_type == 'transform':
                    color = '#FFA500'  # Orange
                elif script_type == 'consumer':
                    color = '#FF6347'  # Tomato
                elif script_type == 'definition':
                    color = '#DDA0DD'  # Plum
                else:
                    color = '#D3D3D3'  # Light gray

                symbol = 'hexagon'
                size = 30

                transformations = details.get('transformations', [])
                hover_text = f"<b>{node_name}</b><br>Type: {script_type}<br>Transformations: {', '.join(transformations) if transformations else 'None'}"

            else:
                color = '#808080'  # Gray
                symbol = 'circle'
                size = 15
                hover_text = f"<b>{node_name}</b><br>Type: {node_type}"

            # Highlight path if specified
            if highlight_path and node_id in highlight_path:
                color = '#FF1493'  # Deep pink
                size = size * 1.5

            node_colors.append(color)
            node_sizes.append(size)
            node_symbols.append(symbol)
            node_text.append(hover_text)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node.get('name', '')[:20] for node in nodes if node['id'] in pos],
            textposition="top center",
            textfont=dict(size=8),
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                symbol=node_symbols,
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])

        fig.update_layout(
            title={
                'text': "Data Lineage Dependency Graph",
                'x': 0.5,
                'xanchor': 'center'
            },
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="🟦 HDFS Path&nbsp;&nbsp;&nbsp;🔷 Hive Table&nbsp;&nbsp;&nbsp;⬡ Script",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.1,
                    xanchor='center', yanchor='top'
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(240,240,240,0.9)'
        )

        # Add legend manually
        legend_traces = [
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=15, color='#90EE90', symbol='hexagon'), name='Source Script'),
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=15, color='#FFA500', symbol='hexagon'), name='Transform Script'),
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=15, color='#FF6347', symbol='hexagon'), name='Consumer Script'),
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=15, color='#FFD700', symbol='square'), name='HDFS Path'),
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=15, color='#4169E1', symbol='diamond'), name='Hive Table'),
        ]

        for trace in legend_traces:
            fig.add_trace(trace)

        return fig

    @staticmethod
    def _hierarchical_layout(G: nx.DiGraph) -> Dict:
        """Calculate hierarchical layout (left-to-right)"""
        try:
            # Use topological sort to create layers
            layers = list(nx.topological_generations(G))
        except:
            # If graph has cycles, use spring layout
            return nx.spring_layout(G)

        pos = {}
        max_width = max(len(layer) for layer in layers) if layers else 1

        for layer_idx, layer in enumerate(layers):
            x = layer_idx / max(len(layers) - 1, 1)  # Normalize to 0-1

            for node_idx, node in enumerate(sorted(layer)):
                y = (node_idx + 0.5) / max(len(layer), 1)  # Normalize to 0-1
                pos[node] = (x, y)

        return pos

    @staticmethod
    def _create_empty_graph() -> go.Figure:
        """Create empty placeholder graph"""
        fig = go.Figure()
        fig.add_annotation(
            text="No lineage data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig

    @staticmethod
    def create_graphviz_diagram(
        lineage_graph: Dict[str, Any],
        output_format: str = "png"
    ) -> Optional[bytes]:
        """
        Create static diagram using Graphviz

        Args:
            lineage_graph: Lineage graph dict
            output_format: Output format ("png", "svg", "pdf")

        Returns:
            Rendered diagram as bytes, or None if Graphviz unavailable
        """
        if not GRAPHVIZ_AVAILABLE:
            logger.warning("Graphviz not available")
            return None

        nodes = lineage_graph.get('nodes', [])
        edges = lineage_graph.get('edges', [])

        # Create directed graph
        dot = graphviz.Digraph(comment='Data Lineage')
        dot.attr(rankdir='LR')  # Left to right
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        dot.attr('edge', fontname='Arial', fontsize='10')

        # Add nodes
        for node in nodes:
            node_id = node['id']
            node_name = node.get('name', node_id)
            node_type = node.get('type', 'unknown')

            if node_type == 'data_location':
                color = 'lightblue'
                shape = 'cylinder'
            elif node_type == 'script':
                details = node.get('details', {})
                script_type = details.get('type', 'unknown')

                if script_type == 'source':
                    color = 'lightgreen'
                elif script_type == 'transform':
                    color = 'orange'
                elif script_type == 'consumer':
                    color = 'salmon'
                else:
                    color = 'lightgray'

                shape = 'box'
            else:
                color = 'white'
                shape = 'ellipse'

            dot.node(node_id, node_name, fillcolor=color, shape=shape)

        # Add edges
        for edge in edges:
            source = edge['source']
            target = edge['target']
            edge_type = edge.get('type', '')
            script = edge.get('script', '')

            label = f"{edge_type}"
            if script:
                label += f"\n{script}"

            if edge_type == 'reads':
                color = 'blue'
            elif edge_type == 'writes':
                color = 'green'
            else:
                color = 'gray'

            dot.edge(source, target, label=label, color=color)

        # Render
        try:
            return dot.pipe(format=output_format)
        except Exception as e:
            logger.error(f"Error rendering Graphviz diagram: {e}")
            return None

    @staticmethod
    def generate_mermaid_diagram(lineage_graph: Dict[str, Any]) -> str:
        """
        Generate Mermaid diagram syntax for lineage

        Args:
            lineage_graph: Lineage graph dict

        Returns:
            Mermaid diagram as string
        """
        nodes = lineage_graph.get('nodes', [])
        edges = lineage_graph.get('edges', [])

        mermaid = ["graph LR"]

        # Add nodes with styling
        for node in nodes:
            node_id = node['id'].replace(':', '_').replace('/', '_')  # Sanitize for Mermaid
            node_name = node.get('name', node_id)[:30]  # Limit length
            node_type = node.get('type', 'unknown')

            if node_type == 'data_location':
                mermaid.append(f"    {node_id}[({node_name})]")
            elif node_type == 'script':
                details = node.get('details', {})
                script_type = details.get('type', 'unknown')
                mermaid.append(f"    {node_id}{{{node_name}<br/>{script_type}}}")
            else:
                mermaid.append(f"    {node_id}[{node_name}]")

        # Add edges
        for edge in edges:
            source = edge['source'].replace(':', '_').replace('/', '_')
            target = edge['target'].replace(':', '_').replace('/', '_')
            edge_type = edge.get('type', '')

            if edge_type == 'reads':
                mermaid.append(f"    {source} -->|reads| {target}")
            elif edge_type == 'writes':
                mermaid.append(f"    {source} -.writes.-> {target}")
            else:
                mermaid.append(f"    {source} --> {target}")

        return "\n".join(mermaid)
