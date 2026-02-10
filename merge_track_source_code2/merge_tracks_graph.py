import plotly.offline as py 
import plotly.graph_objs as go 

from merge_tracks import *

def test():
    trace = go.Scatter( 
        x=[1, 2, 2, 1], 
        y=[3, 4, 3, 4], 
        mode='markers',
        marker=dict(size=[100, 100, 100, 100])
    )

    fig = go.Figure(
        data=[trace],
        layout=go.Layout(
            annotations = [
                dict(
                    ax=1, ay=3, axref='x', ayref='y',
                    x=2, y=4, xref='x', yref='y'
                ),
                # eval("dict(ax=2, ay=3, axref='x', ayref='y', x=1, y=4, xref='x', yref='y')")
            ]
        )
    ) 
    py.plot(fig)


def convert_nx_2_dash():
    merger = merge_tracks()
    merger.TRACK_DATA_FILE = '../tracking reindex/Thầy Hải/out_txts/GH010371.txt'
    merger.FIX_TRACK_ID = 140
    merger.FIX_HAND_SIDE = 1  # 1 trái, 2 phải
    # delta_t: thời gian để tính tỉ lệ activate chuyển tiếp giữa 2 track
    merger.START_TRACKING_TIME_SECONDS = 170
    merger.STOP_TRACKING_TIME_SECONDS = 290
    merger.init()
    graph = merger.build_graph()
    path_length, path_nodes, negative_cycle = bf.bellman_ford(
        graph, source=merger.FIX_TRACK_ID, target= merger.max_track_id + 1, weight="length")
    # path_nodes.pop()
    print(path_length)
    print(path_nodes)
    left_nodes = path_nodes.copy()

    nodes_set = set()
    nodes = []
    edges = []
    nodes_di = {}
    edges_di = {}
    followers_node_di = {}
    followers_edges_di = {} 
    for n in graph.nodes():
        if n >= merger.FIX_TRACK_ID:
            node = {"data": {"id": n, "label": "track #" + str(n)}}
            node['classes'] = "node"
            nodes.append(node)
            nodes_di[n] = node
            nodes_set.add(n)            
            if n == merger.FIX_TRACK_ID or n == merger.max_track_id + 1:
                node['classes'] = "genesis"
    for e in graph.edges():
        source = e[0]
        target = e[1]
        if e[0] in nodes_set and e[1] in nodes_set:
            edge = {'data': {'id': e[0] * 1000 + e[1], 'source': e[0], 'target': e[1]}}
            edge['classes'] = "edge"
            if not followers_node_di.get(target):
                followers_node_di[target] = []
            if not followers_edges_di.get(target):
                followers_edges_di[target] = []
            followers_node_di[target].append(nodes_di[source])
            followers_edges_di[target].append(edge)
            # edges.append(edge)
            for i in range(len(path_nodes) - 1):
                if (e[0] == path_nodes[i] and e[1] == path_nodes[i + 1]):
                    edge['classes'] = "followingEdge"
                    edges.append(edge)
    return nodes, edges, nodes_di, edges_di, followers_node_di, followers_edges_di

