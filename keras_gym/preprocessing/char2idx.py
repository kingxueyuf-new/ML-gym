import string
from collections import OrderedDict

char_2_idx = OrderedDict()
index = 1


def char_to_idx(chr):
    return char_2_idx[chr]


def init_char_2_idx():
    global index
    global char_2_idx
    letters = string.ascii_letters + string.digits + string.punctuation
    for c in letters:
        char_2_idx[c] = index
        index += 1
    print(char_2_idx)


if __name__ == "__main__":
    init_char_2_idx()
    print(char_to_idx('b'))
