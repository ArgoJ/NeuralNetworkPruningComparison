from pyx import *
import torch
import torch.nn as nn
import numpy as np


from NeuralNetwork import Model



COLORS = [
    color.rgb(0.93, 0.8, 0.38),
    color.rgb(0.89, 0.66, 0.41), 
    color.rgb(0.38, 0.51, 0.65),
    color.rgb(0.46, 0.3, 0.47),
    color.rgb(0.31, 0.18, 0.31),
    color.rgb(0.33, 0.6, 0.2),
    color.rgb(0.55, 0.28, 0.15),
    color.rgb(0.18, 0.39, 0.27)
]

STACKED_WIDTH = 4
STACKED_HEIGHT = 4
CORNER_RADIUS = 0.15
STACKED_BEHIND_X_DIFF = STACKED_WIDTH*0.08
STACKED_BEHIND_Y_DIFF = STACKED_HEIGHT*0.08
ELEMENT_WIDTH = STACKED_WIDTH*0.12
ELEMENT_HEIGHT = STACKED_HEIGHT*0.11
OUTLINE_TOP_DISTANCE = STACKED_HEIGHT*0.3
OUTLINE_SIDE_DISTANCE = STACKED_WIDTH*0.15
OUTLINE_BOTTOM_DISTANCE = OUTLINE_SIDE_DISTANCE

SPACE_BETWEEN_LAYER = STACKED_WIDTH*0.18
SPACE_BETWEEN_ARROW = SPACE_BETWEEN_LAYER*0.5

MATRIX_TEXT_DISTANCE = STACKED_HEIGHT*0.05
ARROW_LENGTH = STACKED_WIDTH*0.2




def makeDraws(model: Model):
    curr_x = 0
    curr_y = 0
    prev_features = False
    hidden_layer = 1

    textrunner = text.set(text.LatexRunner)
    c = canvas.canvas(textengine=textrunner)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # print(list(model.named_parameters()))
            in_features = module.in_features
            out_features = module.out_features


            # matrix sizes
            weights_matrix_width = in_features*ELEMENT_WIDTH
            weights_matrix_height = out_features*ELEMENT_HEIGHT

            bias_matrix_width = 1*ELEMENT_WIDTH
            bias_matrix_height = out_features*ELEMENT_HEIGHT 

            outline_width = weights_matrix_width + bias_matrix_width + 3*OUTLINE_SIDE_DISTANCE
            outline_height = bias_matrix_height + OUTLINE_TOP_DISTANCE + OUTLINE_BOTTOM_DISTANCE

            h_width = 1.5*ELEMENT_WIDTH
            h_height = bias_matrix_height


            # input layer
            if not prev_features:
                prev_features = True

                # input matrix size
                input_matrix_width = h_width
                input_matrix_height = in_features*ELEMENT_HEIGHT

                # input matrix
                input_text = r'$x\textsubscript{i}$' # \textsuperscript{(0)}
                input_layer_data = np.ones((1, in_features))
                _get_matrix(
                    c, 
                    input_layer_data, 
                    curr_x, curr_y, 
                    0, 
                    input_matrix_width, 
                    input_matrix_height, 
                    matrix_text=input_text,
                    element_width=input_matrix_width,
                    styles=[style.linewidth.THick]
                )
                curr_x += input_matrix_width + SPACE_BETWEEN_ARROW

                # arrow after input
                _get_arrow(
                    c, 
                    curr_x, curr_y
                )
                curr_x += ARROW_LENGTH + SPACE_BETWEEN_ARROW
                

            # outline
            layer_name = name.split('.')[-1]
            input_layer_text = r'{layer_name}'.format(layer_name=layer_name.replace('_', '\,'))
            _get_outline_rect(
                c, 
                curr_x, curr_y, 
                outline_width, 
                outline_height,
                bias_matrix_height, 
                layer_text=input_layer_text
            )
            curr_x += OUTLINE_SIDE_DISTANCE


            # weight matrix
            weights_text = r'$w\textsubscript{{ji}}\textsuperscript{{({layer})}}$'.format(
                layer=hidden_layer
            )
            weight_data = module.weight.detach().cpu().numpy()
            weight_transposed = np.transpose(weight_data)
            _get_matrix(
                c,
                weight_transposed, 
                curr_x, curr_y, 
                hidden_layer, 
                weights_matrix_width, 
                weights_matrix_height, 
                matrix_text=weights_text
            )
            curr_x += weights_matrix_width + OUTLINE_SIDE_DISTANCE
                

            # bias matrix
            bias_text = r'$b\textsubscript{{i}}\textsuperscript{{({layer})}}$'.format(
                layer=hidden_layer
            )
            bias_data = np.reshape(module.bias.detach().cpu().numpy(), (1, -1))
            _get_matrix(
                c,
                bias_data, 
                curr_x, curr_y, 
                hidden_layer, 
                bias_matrix_width, 
                bias_matrix_height, 
                matrix_text=bias_text
            )
            curr_x += bias_matrix_width + OUTLINE_SIDE_DISTANCE + SPACE_BETWEEN_LAYER


            # h matrix
            h_text = r'$h\textsubscript{{i}}\textsuperscript{{({layer})}}$'.format(
                layer=hidden_layer
            ) if 'output' not in name else r'$\hat{y}\textsubscript{i}$'
            input_data = np.ones((1, in_features))
            h_data = input_data @ weight_transposed + bias_data
            _get_matrix(
                c, 
                h_data,
                curr_x, curr_y, 
                hidden_layer, 
                h_width, 
                h_height, 
                h_text,
                element_width=h_width, 
                styles=[style.linewidth.THick]
            )
            curr_x += h_width + SPACE_BETWEEN_ARROW


            # arrow after layer
            if 'output' not in name:
                _get_arrow(
                    c, 
                    curr_x, curr_y
                )
                curr_x += ARROW_LENGTH + SPACE_BETWEEN_ARROW


            hidden_layer += 1
    return c



#=================================================================================
# ARROW DRAW
#=================================================================================
def _get_arrow(
    c: canvas.canvas,
    x_pt, y_pt,
    arrow_length=ARROW_LENGTH
):
    arrow_head_pt = x_pt + arrow_length
    c.stroke(
        path.path(
            path.moveto(x_pt, y_pt), 
            path.lineto(arrow_head_pt, y_pt)
        ),
        [style.linewidth.THick, deco.earrow.Large]
    )
    return c



#=================================================================================
# LAYER OUTLINE DRAW
#=================================================================================
def _get_outline_rect(
    c: canvas.canvas,
    x_pt, y_pt, 
    outline_width, 
    outline_height, 
    matrix_height, 
    bottom_distance=OUTLINE_BOTTOM_DISTANCE, 
    layer_text=None, 
    corner_radius=CORNER_RADIUS
):
    y_pt -= bottom_distance + matrix_height / 2
    c.stroke(
        _round_rect(x_pt, y_pt, outline_width, outline_height, corner_radius), 
            [style.linewidth.THick]
    )
    if layer_text is not None:
        x_pt += outline_width / 2
        y_pt -= MATRIX_TEXT_DISTANCE
        c = _get_layer_text(c, layer_text, x_pt, y_pt)
    return c



#=================================================================================
# MATRIX DRAW
#=================================================================================
def _get_matrix(
    c: canvas.canvas,
    input_data: torch.Tensor, 
    x_pt, y_pt, 
    layer: int,
    matrix_width, 
    matrix_height, 
    matrix_text=None,
    element_width=ELEMENT_WIDTH, 
    element_height=ELEMENT_HEIGHT, 
    corner_radius=CORNER_RADIUS,
    styles=[style.linewidth.Thick]
):
    y_pt -= matrix_height / 2
    c.insert(_get_matrix_colored_inside(
        x_pt, y_pt, 
        input_data, 
        layer, 
        element_width=element_width,
        element_height=element_height
    ))
    c.stroke(
        _round_rect(
            x_pt, y_pt, 
            matrix_width, 
            matrix_height, 
            corner_radius), 
        styles
    )
    if matrix_text is not None:
        x_pt += matrix_width / 2
        y_pt += MATRIX_TEXT_DISTANCE + matrix_height
        c = _get_matrix_text(
            c,
            matrix_text,
            x_pt, y_pt
        )
    return c


def _get_matrix_colored_inside(
    x_pt, y_pt, 
    input_data: torch.Tensor, 
    layer: int, 
    element_width=ELEMENT_WIDTH, 
    element_height=ELEMENT_HEIGHT, 
    corner_radius=CORNER_RADIUS,
    colors=COLORS
) -> canvas.canvas:
    input_size = input_data.shape
    rows = input_size[0]
    cols = input_size[1]

    color_values = _get_flipud_color_values(input_data) 

    row_pruned = np.all(color_values == 1., axis=1) if rows!=1 else [False,]
    col_pruned = np.all(color_values == 1., axis=0)
    
    cl = canvas.canvas([canvas.clip(_round_rect(x_pt, y_pt, rows*element_width, cols*element_height, corner_radius))])
    for nx in range(rows):
        for ny in range(cols):
            c_val = color_values[nx, ny]
            if c_val!=1.0:
                rect_color = color.gray(c_val)
            elif c_val==1.0 and col_pruned[ny] and not row_pruned[nx]:
                rect_color = colors[layer]
            elif c_val==1.0 and row_pruned[nx] and not col_pruned[ny]:
                rect_color = colors[layer-1]
            else:
                rect_color = colors[0]

            cl.stroke(
                path.rect(
                    x_pt + element_width*nx, 
                    y_pt + element_height*ny, 
                    element_width, 
                    element_height
                ),
                [deco.filled([rect_color])]
            )
    return cl


def _get_single_element(cl, x_pt, y_pt, nx, ny, element_width, element_height, color_values, colors, layer, row_pruned, col_pruned):
    c_val = color_values[nx, ny]
    if c_val!=1.0:
        rect_color = color.gray(c_val)
    elif c_val==1.0 and col_pruned[ny] and not row_pruned[nx]:
        rect_color = colors[layer-1]
    elif c_val==1.0 and row_pruned[nx] and not col_pruned[ny]:
        rect_color = colors[layer]
    else:
        rect_color = colors[0]

    cl.stroke(
        path.rect(
            x_pt + element_width*nx, 
            y_pt + element_height*ny, 
            element_width, 
            element_height
        ),
        [deco.filled([rect_color])]
    )



#=================================================================================
# STACKED LAYERS
#=================================================================================
def _layer_stacked(
    c: canvas.canvas, 
    curr_x, curr_y, 
    features, 
    input_data: torch.Tensor=None, 
    pruned_color: color=COLORS[0],
    rect_width=STACKED_WIDTH, 
    rect_height=STACKED_HEIGHT, 
    corner_radius=CORNER_RADIUS,
    x_diff=STACKED_BEHIND_X_DIFF, 
    y_diff=STACKED_BEHIND_Y_DIFF
):
    color_values = [0.9 for _ in range(features)] if input_data is None else _get_color_values(input_data)
    first_node_color = color.gray(color_values[0]) if color_values[0]!=1.0 else pruned_color
    c.stroke(
        _round_rect(
            curr_x, curr_y,
            rect_width,
            rect_height, 
            corner_radius
        ), 
        [deco.filled([first_node_color]), style.linewidth.Thick]
    )
    for i in range(1, features):
        node_i_color = color.gray(color_values[i]) if color_values[i]!=1.0 else pruned_color
        curr_x += x_diff
        curr_y += y_diff
        c.stroke(
            _rect_on_top(
                curr_x, curr_y,
                rect_width,
                rect_height, 
                corner_radius,
                x_diff, 
                y_diff
            ), 
            [deco.filled([node_i_color]), style.linewidth.Thick]
        )
    curr_x += STACKED_WIDTH*1.18
    return c, curr_x, curr_y



#=================================================================================
# TEXTS
#=================================================================================
def _get_matrix_text(c: canvas.canvas, matrix_text, x_pt, y_pt):
    c.text(
        x_pt, y_pt, 
        matrix_text, 
        [text.halign.center, text.valign.bottom, text.size.huge]
    )
    return c


def _get_layer_text(c: canvas.canvas, layer_text, x_pt, y_pt):
    c.text(
        x_pt, y_pt, 
        layer_text, 
        [text.halign.center, text.valign.top, text.size.huge]
    )
    return c

    
def _get_node_text(c: canvas.canvas, node_tex, x_pt, y_pt):
    c.text(
        x_pt, y_pt, 
        node_tex, 
        [trafo.rotate(50), text.halign.center, text.valign.bottom, text.size.huge]
    )
    return c



#=================================================================================
# CALCULATIONS
#=================================================================================
def _get_color_values(input: np.ndarray):
    max_input = np.max(np.abs(input))
    return 1 - 0.75*(np.abs(input) / max_input) if input.size!=1 else np.array([[0.9]])


def _get_flipud_color_values(input: np.ndarray):
    return _get_color_values(np.fliplr(input))



#=================================================================================
# RECTANGULARS
#=================================================================================
def _round_rect(x_pt, y_pt, width, height, radius) -> path.path:
    # low left corner
    x_l_low = x_pt
    y_l_low = y_pt
    # low right corner
    x_r_low = x_pt + width
    y_r_low = y_pt
    # top right corner
    x_r_top = x_pt + width
    y_r_top = y_pt + height
    # top left corner
    x_l_top = x_pt 
    y_l_top = y_pt + height
    return path.path(
        path.moveto(x_l_low + radius, y_l_low), 
        path.lineto(x_r_low - radius, y_r_low), 
        path.arc(x_r_low - radius, y_r_low + radius, radius, 270, 0), 
        path.lineto(x_r_top, y_r_top - radius), 
        path.arc(x_r_top - radius, y_r_top - radius, radius, 0, 90), 
        path.lineto(x_l_top + radius, y_l_top), 
        path.arc(x_l_top + radius, y_l_top - radius, radius, 90, 180), 
        path.lineto(x_l_low, y_l_low + radius), 
        path.arc(x_l_low + radius, y_l_low + radius, radius, 180, 270), 
        path.closepath()
    )



def _rect_on_top(x_pt, y_pt, width, height, radius, x_diff, y_diff) -> path.path:
    # low right corner
    x_r_low = x_pt + width
    y_r_low = y_pt
    # top right corner
    x_r_top = x_pt + width
    y_r_top = y_pt + height
    # top left corner
    x_l_top = x_pt 
    y_l_top = y_pt + height
    return path.path(
        path.moveto(x_r_low - x_diff, y_r_low),
        path.lineto(x_r_low - radius, y_r_low), 
        path.arc(x_r_low - radius, y_r_low + radius, radius, 270, 0), 
        path.lineto(x_r_top, y_r_top - radius), 
        path.arc(x_r_top - radius, y_r_top - radius, radius, 0, 90), 
        path.lineto(x_l_top + radius, y_l_top), 
        path.arc(x_l_top + radius, y_l_top - radius, radius, 90, 180),
        path.lineto(x_l_top, y_l_top - y_diff),
        path.lineto(x_r_top - x_diff - radius, y_r_top - y_diff), 
        path.arcn(x_r_top - x_diff - radius, y_r_top - y_diff - radius, radius, 90, 0),
        path.closepath()
    )





