import sys, os
import xml.etree.ElementTree as ET
import numpy as np
from skimage.io import imsave
from skimage.draw import line
import scipy.ndimage as ndimage

import warnings
warnings.filterwarnings('ignore')

def parse_inkml(inkml_file_abs_path):
    if inkml_file_abs_path.endswith('.inkml'):
        tree = ET.parse(inkml_file_abs_path)
        root = tree.getroot()
        doc_namespace = "{http://www.w3.org/2003/InkML}"
        # Stores traces with their corresponding id
	    # MM: multiple all integers and floats by 10K
        traces_all_list = [{'id': trace_tag.get('id'),
                            'coords': [[round(float(axis_coord) * 10000) \
                                            for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') \
                                        else [round(float(axis_coord) * 10000) \
                                            for axis_coord in coord.split(' ')] \
                                    for coord in (trace_tag.text).replace('\n', '').split(',')]} \
                                    for trace_tag in root.findall(doc_namespace + 'trace')]

        # convert in dictionary traces_all by id to make searching for references faster
        traces_all = {}
        for t in traces_all_list:
            traces_all[t["id"]] = t["coords"]
        return traces_all
    else:
        print('File ', inkml_file_abs_path, ' does not exist !')
        return {}

# get traces of data from inkml file and convert it into bmp image
def get_traces_data(traces_dict, id_set = None):
    # Accumulates traces_data of the inkml file
    traces_data_curr_inkml=[]
    if id_set == None:
        id_set = traces_dict.keys()
    # this range is specified by values specified in the lg file
    for i in id_set: # use function for getting the exact range
        traces_data_curr_inkml.append(traces_dict[i])
    return traces_data_curr_inkml

def get_min_coords(traces):
    x_coords = [coord[0] for coord in traces]
    y_coords = [coord[1] for coord in traces]
    min_x_coord=min(x_coords)
    min_y_coord=min(y_coords)
    max_x_coord=max(x_coords)
    max_y_coord=max(y_coords)
    return min_x_coord, min_y_coord, max_x_coord, max_y_coord

def shift_trace(traces, min_x, min_y):
    # shift pattern to its relative position
    shifted_trace = [[coord[0] - min_x, coord[1] - min_y] for coord in traces]
    return shifted_trace

def scaling(traces, scale_factor=1.0):
    # Scaling: Interpolates a pattern so that it fits into a box with specified size
    interpolated_trace = []
    # coordinate convertion to int type necessary
    interpolated_trace = [[round(coord[0] * scale_factor), round(coord[1] * scale_factor)] for coord in traces]
    return interpolated_trace

def center_pattern(traces, max_x, max_y, box_axis_size):
    # Centering: Shifts the pattern so that it is centered in the box
    x_margin = int((box_axis_size - max_x) / 2)
    y_margin = int((box_axis_size - max_y) / 2)
    return shift_trace(traces, min_x= -x_margin, min_y= -y_margin)

def draw_pattern(traces,pattern_drawn, box_axis_size):
    # if only single point
    if len(traces) == 1:
            x_coord = traces[0][0]
            y_coord = traces[0][1]
            pattern_drawn[y_coord, x_coord] = 0.0 # draw black
    else:
        # if more than one point iterate through list of trace endpoints
        for pt_idx in range(len(traces) - 1):
                # Draw line between the two endpoints of the trace
                linesX = linesY = []
                oneLineX, oneLineY = line(r0=traces[pt_idx][1], c0=traces[pt_idx][0],
                                              r1=traces[pt_idx + 1][1], c1=traces[pt_idx + 1][0])

                linesX = np.concatenate(
                    [oneLineX, oneLineX, oneLineX+1]) # We can use this to draw a thicker line
                linesY = np.concatenate(
                    [oneLineY+1, oneLineY, oneLineY])

                # Ensure that the line is within the box (set to 0 or border axis size if not)
                linesX[linesX < 0] = 0
                linesX[linesX >= box_axis_size] = box_axis_size-1

                linesY[linesY < 0] = 0
                linesY[linesY >= box_axis_size] = box_axis_size-1

                pattern_drawn[linesX, linesY] = 0.0
    return pattern_drawn


def convert_to_imgs(traces_data, box_axis_size, lg_dict):
    """

    :param traces_data: All data from one trace
    :param box_axis_size: Size of the box in which the pattern is drawn
    :param lg_dict: Dictionary of the parsed lg file in format {stroke_number: symbol_drawn} or
    {stroke_number: "save"} if symbol is not finished with this stroke
    :return: Drawn pattern
    """
    pattern_drawn = np.ones(shape=(box_axis_size, box_axis_size), dtype=np.float32)
    # Special case of inkml file with zero trace (empty)
    if len(traces_data) == 0:
        return np.matrix(pattern_drawn * 255, np.uint8)

    # mid coords needed to shift the pattern to its relative position
    min_x, min_y, max_x, max_y = get_min_coords([item for sublist in traces_data for item in sublist]  )

    # trace dimensions
    trace_height, trace_width = max_y - min_y, max_x - min_x
    if trace_height == 0:
        trace_height += 1
    if trace_width == 0:
        trace_width += 1

    # KEEP original size ratio
    trace_ratio = (trace_width) / (trace_height)
    box_ratio = box_axis_size / box_axis_size

    # Set \"rescale coefficient\" magnitude
    if trace_ratio < box_ratio:
        scale_factor = ((box_axis_size-1) / trace_height)
    else:
        scale_factor = ((box_axis_size-1) / trace_width)

    # Create empty array incase the lg_dict entry equals "save"
    saved_array = []
    label_lines = []
    for index, traces_all in enumerate(traces_data):
        # shift pattern to its relative position
        shifted_trace = shift_trace(traces_all, min_x=min_x, min_y=min_y)
        # Scaling: Interpolates a pattern so that it fits into a box with specified size with Linear interpolation
        try:
            scaled_trace = scaling(shifted_trace,scale_factor)
        except Exception as e:
            print(e)
            print('This data is corrupted - skipping.')

        # Centering: Shifts the pattern so that it is centered in the box
        centered_trace = center_pattern(scaled_trace, max_x=trace_width*scale_factor, max_y=trace_height*scale_factor, box_axis_size=box_axis_size-1)

        # Draw pattern on the background
        pattern_drawn = draw_pattern(centered_trace, pattern_drawn,box_axis_size=box_axis_size)
        label = lg_dict[index]
        # Save the drawn pattern if the lg_dict entry equals "save"
        if label == "save":
            if len(saved_array) == 0:
                saved_array = np.array(centered_trace)
            else:
                saved_array = np.vstack((saved_array, np.array(centered_trace))) # shape (n, 2)
        # If symbol is finished with this stroke, save bounding box and reset saved_array
        else:
            if len(saved_array) > 0:
                final_array = np.vstack((saved_array, np.array(centered_trace))) # shape (n, 2)
                saved_array = []
            else:
                final_array = np.array(centered_trace)
            label_lines.append(create_bounding_box(final_array, box_axis_size, label))
    #advanced-dataset-generation = np.matrix(pattern_drawn * 255, np.uint8) # Uncomment to see bounding boxes (also uncomment in create_bounding_box)
    #plt.imshow(advanced-dataset-generation)
    #plt.show()
    return np.matrix(pattern_drawn * 255, np.uint8), label_lines


def create_bounding_box(final_array, dim, label):
    min_x_coord = max(min(final_array[:, 0]) - 1, 0)
    min_y_coord = max(min(final_array[:, 1]) - 1, 0)
    max_x_coord = min(max(final_array[:, 0]) + 1, dim-1)
    max_y_coord = min(max(final_array[:, 1]) + 1, dim-1)
    width = max_x_coord - min_x_coord
    height = max_y_coord - min_y_coord
    # remove all whitespace from label string
    label = label.replace(" ", "")
    #plt.gca().add_patch(
    #    Rectangle((min_x_coord, min_y_coord), width, height, fill=False, edgecolor='red', linewidth=1))
    return [min_x_coord, min_y_coord, max_x_coord, max_y_coord, label]


if __name__ == '__main__':
    """ 2 usages :
    convertInkmlToImg.py file.inkml (dim) (padding) (outDir)
    convertInkmlToImg.py folder (dim) (padding) (outDir)

    Example
    python3 convertInkmlToImg.py ../../../DB_CRHOME/task2-validation-isolatedTest2013b 28 2 
    """
    if len(sys.argv) < 2:
        print('\n + Usage:', sys.argv[0], ' (file|folder) dim padding outdir')
        print('\t+ {:<20} - required str'.format("(file|folder)"))
        print('\t+ {:<20} - optional int (def = 300)'.format("dim"))
        print('\t+ {:<20} - optional int (def =  0)'.format("padding"))
        exit()
    else:
        if os.path.isfile(sys.argv[1]):
            FILES = [sys.argv[1]]
        else:
            from glob import glob
            if sys.argv[1][-1] != os.sep: sys.argv[1] += os.sep
            FILES = glob(sys.argv[1]+os.sep+"*.inkml")
        
        folder_name = sys.argv[1].split(os.sep)[-2]

        # save_path = "data_png_" + folder_name if len(sys.argv) < 5 else sys.argv[4] + "data_png_" #+ folder_name
        save_path = "archive" + os.sep + "formulas"

        # save labels path
        save_path_labels = save_path
        # save_path_labels = "data_labels_" + folder_name if len(sys.argv) < 5 else sys.argv[4]+"data_labels_" #+ folder_name
        # if not os.path.exists(save_path_labels):
        #     os.makedirs(save_path_labels)

        print("to : " + save_path)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        dim = 300 if len(sys.argv) < 3 else int(sys.argv[2])
        padding = 0 if len(sys.argv) < 4 else int(sys.argv[3])
        lg_folder = "Test2012LG" if len(sys.argv) < 5 else sys.argv[4]

        print("Starting inkml to png conversion on {} file{}\n".format(
            len(FILES), "s" if len(FILES) > 1 else ""
            ))
        
        # Create label txt file
        label_txt = open(save_path_labels + os.sep + 'label' + ".txt", "a")

        for idx, file in enumerate(FILES):

            img_path = os.sep.join(file.split(os.sep)[:-1])
            img_name = file.split(os.sep)[-1]
            img_basename = ".".join(img_name.split(".")[:-1])

            if os.path.isfile(save_path + os.sep + img_basename + '.png'): continue

            if not os.path.isfile(img_path + os.sep + img_name):
                print("\n\nInkml file not found:\n\t{}".format(img_path + os.sep + img_name))
                exit()

            lg_file = lg_folder + os.sep + img_basename + ".lg"
            if not os.path.isfile(lg_file):
                print("\n\nLG file not found:\n\t{}".format(lg_file))
                exit()

            with open(lg_file, 'r') as f:
                # Read all lines in the file
                lines = f.readlines()
            # Read everything from lg file between #Nodes: and #Edges:
            copy_nodes = False
            copy_edges = False
            LG_Nodes = []
            LG_Edges = []
            for file_line in lines:
                # Remove leading and trailing whitespace from the line
                file_line = file_line.strip()
                # Skip empty lines
                if not file_line:
                    continue
                if copy_edges:
                    my_line = file_line.split(",")
                    my_line = [s.replace(" ", "") for s in my_line]
                    if my_line[3] == "*":  # if the stroke is of the same character
                        LG_Edges.append([my_line[1], my_line[2]])
                if file_line.startswith("# Edges:"):
                    copy_nodes = False
                    copy_edges = True
                if copy_nodes:
                    LG_Nodes.append(file_line)
                if file_line.startswith("# Nodes:"):
                    copy_nodes = True

            # Get symbol from LG list in dict with key = stroke num (where it was last executed)
            lg_node_dict = {}
            lg_edge_dict = {}
            for edge_line in LG_Edges:
                prev_stroke, next_stroke = edge_line[0], edge_line[1] # sa
                if prev_stroke in lg_edge_dict:
                    lg_edge_dict[prev_stroke].append(next_stroke)
                else:
                    lg_edge_dict[prev_stroke] = [next_stroke]

            for i, lg in enumerate(LG_Nodes):
                symbol = lg.split(",")[2].replace(" ", "")
                stroke_num = lg.split(",")[1].replace(" ", "")
                if i < len(LG_Nodes) - 1:
                    # if next symbol is the same (also check strokes via edge_dict)
                    next_ = LG_Nodes[i + 1].split(",")[2].replace(" ", "")
                    if next_ == symbol and stroke_num in lg_edge_dict:
                        if LG_Nodes[i + 1].split(",")[1].replace(" ", "") in lg_edge_dict[stroke_num]:
                            # mark as same symbol
                            lg_node_dict[i] = "save"
                        else:
                            lg_node_dict[i] = symbol
                    else:
                        lg_node_dict[i] = symbol
                else:
                    lg_node_dict[i] = symbol

            traces = parse_inkml(img_path + os.sep + img_name)

            selected_tr = get_traces_data(traces)
            im, label_lines = convert_to_imgs(selected_tr, dim, lg_node_dict)

            # initialize img file name
            label_txt.write(save_path + os.sep + img_basename + ".png")

            # 2d array to string with comma between each element and space between arrays
            label_txt.write(" " + " ".join([",".join([str(x) for x in line]) for line in label_lines]))
            label_txt.write('\n')

            if padding > 0:
                im = np.lib.pad(im, (padding, padding), 'constant', constant_values=255)
            
            im = ndimage.gaussian_filter(im, sigma=(.5, .5), order=0)

            imsave(save_path + os.sep + img_basename + '.png',im)

            print("\t\t\rfile: {:>10} | {:>6}/{:}".format(img_basename, idx+1, len(FILES)), end="")

        # Close label_txt file
        label_txt.close()

    print("\n\nFinished")
