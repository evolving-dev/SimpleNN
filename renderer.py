import PIL
from PIL import Image, ImageDraw, ImageFont

def get_node_color(value):
    if value > 1:
        return (255, 255, 255)
    if value < -1:
        return (0, 0, 0)

    value = value / 2 + 0.5 #Convert from range [-1, 1] to [0, 1]

    value = round(value * 255) #Convert to color

    return 3 * (value, )

def get_text_color(color):
    if sum(color) < 384:
        return (255, 255, 255)
    return (0, 0, 0)

def render_nn(network, dimensions = [1024, 1024], inputs = None, background_color = (0, 0, 0, 0), input_labels = [], output_labels = [], line_width = 1):
    im = Image.new("RGBA", tuple(dimensions), tuple(background_color))
    draw = ImageDraw.Draw(im)

    col_count = network.layer_count
    col_width = dimensions[0] // network.layer_count
    subcol_width = col_width // 3 #Diameter of node circle, draw circle in subcolumn 2 of 3

    for layer in range(col_count):
        row_count = network.layers[layer]
        row_height = dimensions[1] // row_count
        subrow_height = row_height // 3
        if inputs != None:

            font = ImageFont.truetype("PTSans-Regular.ttf", subrow_height // 8 * 5)
            
            data = inputs.copy()

            if layer != 0:
                for n_layer in range(layer + 1):
                    data = network.get_layer_output(data, n_layer)
            
        for node in range(row_count):
            
            if inputs != None:
                color = get_node_color(data[node])
            else:
                color = (255, 255, 255)

            text_color = get_text_color(color)

            max_size = max(subcol_width, subrow_height)
            size = min(subcol_width, subrow_height)

            x = layer * col_width + subcol_width
            y = node * row_height + subrow_height

            #Proper centering
            if subcol_width > subrow_height:
                x += 0.5 * (max_size - size)
            elif subcol_width < subrow_height:
                y += 0.5 * (max_size - size)

            #Node connections
            if layer + 1 < network.layer_count:
                for n, next_layer_neuron in enumerate(network.neurons[layer + 1]):
                    node_connection = next_layer_neuron.input_weights[node]
                    
                    if node_connection == 0:
                        continue
                    
                    connection_color = get_node_color(node_connection)

                    x1 = x + size // 2 #Start at node base
                    y1 = y + size // 2

                    next_row_count = network.layers[layer + 1]
                    next_row_height = dimensions[1] // next_row_count
                    next_subrow_height = next_row_height // 3

                    x2 = (layer + 1) * col_width + subcol_width + size // 2
                    y2 = n * next_row_height + next_subrow_height + size // 2
                    

                    next_max_size = max(subcol_width, next_subrow_height)
                    next_size = min(subcol_width, next_subrow_height)

                    #Proper centering
                    if subcol_width > next_subrow_height:
                        x2 += 0.5 * (next_max_size - next_size)
                    elif subcol_width < next_subrow_height:
                        y2 += 0.5 * (next_max_size - next_size)

                    draw.line((x1, y1, x2, y2), fill = connection_color, width = line_width)
            
            draw.ellipse((x, y, x + size, y + size), fill=color)

            

            if inputs != None:
                node_label = str(round(data[node], 2))
                if node_label.startswith("0."):
                    node_label = node_label[1:]
                elif node_label.startswith("-0."):
                    node_label = "-" + node_label[2:]
                    
                draw.text((x + size // 2, y + size // 2), node_label, font=font, fill=text_color, anchor="mm")

            if layer == 0 and len(input_labels) > node:
                label_font = ImageFont.truetype("PTSans-Regular.ttf", subrow_height)
                draw.text((subcol_width // 4 * 3 - size // 2, y + size // 2), input_labels[node], font=label_font, fill=get_text_color(background_color), anchor="mm")
            elif layer == network.layer_count - 1 and len(output_labels) > node:
                label_font = ImageFont.truetype("PTSans-Regular.ttf", subrow_height)
                draw.text((dimensions[0] - (subcol_width // 4 * 3 - size // 2), y + size // 2), output_labels[node], font=label_font, fill=get_text_color(background_color), anchor="mm")

    return im
