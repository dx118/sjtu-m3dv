import os
import json


class PathManager:
    def __init__(self, cfg_path=None):
        self.environ = parse_environ(cfg_path)

    @property
    def base(self):
        return self.environ['DATASET']

    @property
    def info(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(self.base, 'train_val.csv'))
        return df

    @property
    def nodule_path(self):
        return os.path.join(self.base, 'nodule')


def parse_environ(cfg_path=None):
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "ENVIRON")
    assert os.path.exists(cfg_path), "`ENVIRON` does not exists."
    with open(cfg_path) as f:
        environ = json.load(f)
    return environ


class PathManager2:
    def __init__(self, cfg_path=None):
        self.environ = parse_environ(cfg_path)

    @property
    def base(self):
        return self.environ['DATASET']

    @property
    def info(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(self.base, 'sampleSubmission.csv'))
        return df

    @property
    def nodule_path(self):
        return os.path.join(self.base, 'nodule')


PATH = PathManager()
PATH2 = PathManager2()