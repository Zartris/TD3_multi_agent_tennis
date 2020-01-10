from argparse import Namespace


def log_hyper_para(file, args: Namespace):
    args_str = str(args)[10:-1]
    args_str.replace("=", " = ")
    arg_list = args_str.split(", ")
    with open(str(file), "a+") as f:
        f.write("## Hyper parameters used:\n")
        for arg in arg_list:
            f.write("\t" + arg + "\n")


def file_append(file, s):
    with open(str(file), "a+") as f:
        f.write(s)
