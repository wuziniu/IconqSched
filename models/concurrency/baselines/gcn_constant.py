DATAPATH = "/Users/ziniuw/Desktop/research/perf_prediction/workload-performance/pmodel_data/job"
CONFIGPATH = (
    "/Users/ziniuw/Desktop/research/perf_prediction/workload-performance/config.ini"
)


class Arguments:
    def __init__(self):
        self.cuda = False
        self.seed = 42
        self.epochs = 200
        self.lr = 1e-2
        self.weight_decay = 5e-4
        self.hidden = 128
        self.node_dim = 64
        self.dropout = 0.1
        self.save_best = True


args = Arguments()
