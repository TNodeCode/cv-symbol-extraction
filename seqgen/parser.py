import numpy as np


def read_label_file(filename):
    with open(filename, "r") as f:
        content = f.read()
    return content

def parse_label_file(content):
    lines = content.split("\n")
    formulas = []
    boxes = []
    for idx_line in range(len(lines)):
        assert lines[idx_line].count("$") == 2, f"Line {idx_line}: There need to be exactly two $ symbols in a line"
        assert lines[idx_line][0] == "$", f"Line {idx_line}: Line The first symbol must be a $ symbol"
        # Find the position of the second dollar symbol
        end = lines[idx_line][1:].index("$")
        formula = lines[idx_line][1:end+1]
        formulas.append(formula)
        suffix = lines[idx_line][end+3:]
        coords = suffix.split(" ")
        assert len(coords) % 5 == 0, f"Line {idx_line}, formula {formula}: There must be 5n items in the coordinates string"
        boxes.append(np.array(coords, dtype=float).reshape(-1, 5))
    return formulas, boxes

def parse_formula(s, keys):
    lst = []
    pos = 0
    while pos < len(s):
        found = False
        for k in keys:
            if s[pos:].startswith(' '):
                pos += 1
                found=True
                break
            if s[pos:].startswith(k):
                l = len(k)
                lst.append(s[pos:pos+l])
                pos += l
                found=True
                break
        if not found:
            raise Exception(f"${s[pos:]}$ could not be parsed")
    return lst