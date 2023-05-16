import numpy as np

from graphviz import Digraph
from torch import nn


from NeuralNetwork import Model




def makeGraph(
    model: Model, 
    show_source: bool = False,
    rannksep: str = '1.0',
    arrowsize: str = '0.5',
    main_color: str = '#1f77b4',
    sub_color: str = '#d62728',
    fontsize: str = '18',
) -> Digraph:
    """
    """
    dot = Digraph('G', comment='Feed forward neural network')
    dot.graph_attr.update(
        rankdir='LR', 
        ranksep=rannksep, 
        splines='false'
    )
    dot.edge_attr.update(
        arrowsize=arrowsize
    )

    layer_nodes_list: list[str] = []
    layer_index = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Adding input layer nodes
            if layer_index==0:
                with dot.subgraph(name=f'cluster_{layer_index}') as cluster:
                    cluster.attr(label='input layer', color='invis', fontsize=fontsize)
                    nodes_list = _addNodes(
                        cluster, 
                        module.in_features, 
                        layer_index, 
                        main_color=main_color, 
                        sub_color=sub_color
                    )
                    layer_nodes_list.append(nodes_list)
                layer_index += 1

            # Get widths of connections and nodes
            weight_widths, bias_widths = _getWidhts(module)

            # Adding rest of the nodes
            layer_name = name.split('.')[-1].replace('_', ' ')
            with dot.subgraph(name=f'cluster_{layer_index}') as cluster:
                cluster.attr(label=layer_name, color='invis', fontsize=fontsize)
                nodes_list = _addNodes(
                    cluster, 
                    module.out_features, 
                    layer_index,
                    main_color=main_color,
                    sub_color=sub_color,
                    bias_widths=bias_widths,
                    in_group=(module.out_features!=1)
                )
                layer_nodes_list.append(nodes_list)

            # Adding edges
            for k, node_in in enumerate(layer_nodes_list[layer_index-1]):
                for l, node_out in enumerate(layer_nodes_list[layer_index]):
                    if weight_widths[l, k] != 0.5:
                        dot.edge(node_in, node_out, penwidth=str(weight_widths[l, k]), color=main_color)
                    else:
                        dot.edge(node_in, node_out, penwidth='1.5', color=sub_color)
            layer_index += 1
    if show_source:
        print(dot.source)
    return dot
    


def _addNodes(
    dot: Digraph, 
    length: int, 
    layer_index: int, 
    main_color: str,
    sub_color: str,
    bias_widths: np.ndarray = None, 
    in_group: bool = False,
):
    if bias_widths is None:
        bias_widths = np.ones(length)

    nodes = [f'{layer_index}_{node+1}' for node in range(length)]
    for node in range(length):
        if bias_widths[node]!=0.5:
            dot.node(
                nodes[node], 
                label=str(node+1), 
                shape='circle', 
                penwidth=str(bias_widths[node]), 
                color=main_color,
                group=str(node) if in_group else '',
            )
        else:
            dot.node(
                nodes[node], 
                label=str(node+1), 
                shape='circle', 
                penwidth='1.5', 
                color=sub_color, 
                group=str(node) if in_group else '',
            )
    return nodes



def _getWidhts(module: nn.Linear, max_width = 2.5, min_width = 0.5):
    def _get_widths(input: np.ndarray, max_width, min_width):
        abs_input = np.abs(input)
        max_input = np.max(abs_input)
        return (max_width-min_width) * abs_input / max_input + min_width
    
    weight = module.weight.detach().cpu().numpy()
    bias = module.bias.detach().cpu().numpy()

    weight_widths = _get_widths(weight, max_width, min_width)
    bias_widths = _get_widths(bias, max_width, min_width)
    return weight_widths, bias_widths