import helper.misc as hmisc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dir', '-d', type=str)

# Parses the input
p_args = parser.parse_args()
dir_files = p_args.dir

print_shape_obj = hmisc.PrintFileShape(dir_files)
res = print_shape_obj.print_file_shape()
print("Count of shapes")
s = 0
s_slice = 0
for k, v in res.items():
    print(k, v)
    s += int(v)
    s_slice += int(k[-1]) * int(v)

print('\n Total number of files ', s)
print('\n Total number of slices ', s_slice)

# Cant I select a slice..?
