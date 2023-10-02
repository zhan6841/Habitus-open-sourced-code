import sys
from utils import parse_logs

if __name__ == '__main__':
    if(len(sys.argv) != 4):
        print("Usage: %s [logfile] [epoch_begin] [epoch_end]" % (sys.argv[0]))
        exit(0)

    value_list = parse_logs(sys.argv[1])
    print("%.6f" % (sum(value_list[-1][int(sys.argv[2]):int(sys.argv[3])])))