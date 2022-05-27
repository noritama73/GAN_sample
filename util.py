import glob, re, sys


def find_best_model(max_epoch: int):
    tmp_max_epoch = 0
    ok = False
    files = glob.glob("./*", recursive=True)
    for file in files:
        if re.search("Generator_epoch_\d+\.pth", file):
            tmp = int(re.sub(r"\D", "", file))
            if (tmp > tmp_max_epoch) and (tmp < max_epoch):
                ok = True
                tmp_max_epoch = tmp

    return "Generator_epoch_{}_.pth".format(tmp_max_epoch), ok


if __name__ == "__main__":
    args = sys.argv
    print(find_best_model(int(args[1])))
