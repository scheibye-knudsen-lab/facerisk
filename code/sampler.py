import os
import pickle
import numpy as np


class SampleManager:
    def __init__(self, filename=None):
        self.filename = filename
        self.clear()

    def clear(self):
        self.sample_xs = []
        self.sample_ys = []
        self.sample_keys = []

    def load_samples(self):
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as file:
                print(self.filename)
                self.sample_keys, self.sample_xs, self.sample_ys = pickle.load(file)

                self.sample_xs = list(self.sample_xs)
                self.sample_ys = list(self.sample_ys)
                self.sample_keys = list(self.sample_keys)
                print("Loaded samples:", len(self.sample_xs), self.sample_xs[0].shape, self.sample_xs[0].dtype)
        else:
            print("*** No samples exist!", self.filename)

    def save_samples(self):
        with open(self.filename, "wb") as file:
            pickle.dump(
                (self.sample_keys, self.sample_xs, self.sample_ys),
                file,
                pickle.HIGHEST_PROTOCOL,
            )
            print("Saved samples:", np.asarray(self.sample_xs).shape)

    def get(self):
        return self.sample_xs, self.sample_ys, self.sample_keys

    def count(self):
        return len(self.sample_keys)

    def add(self, x, y, key=None):
        self.sample_xs.append(x)
        self.sample_ys.append(y)

        if key is None:
            key = "auto:" + str(len(self.sample_xs))

        self.sample_keys.append(key)

        return key

    def remove(self, key):
        pos = self.sample_keys.index(key)
        self.remove_at(pos)

    def remove_at(self, pos):
        del self.sample_xs[pos]
        del self.sample_ys[pos]
        del self.sample_keys[pos]

    def get_at(self, pos):
        return self.sample_keys[pos], self.sample_xs[pos], self.sample_ys[pos]

    def find_sample(self, key):
        try:
            pos = self.sample_keys.index(key)
            return pos, self.sample_xs[pos], self.sample_ys[pos]
        except ValueError:
            return None, None, None

    def map_x_by_y(self):
        y_groups = {}
        for pos in range(len(self.sample_ys)):
            yv = self.sample_ys[pos]
            if yv not in y_groups:
                y_groups[yv] = []

            y_groups[yv].append(self.sample_xs[pos])

        return y_groups

    def get_by_y(self, yval):
        xx, kk = [], []

        for idx in range(len(self.sample_xs)):
            if self.sample_ys[idx] == yval:
                xx.append(self.sample_xs[idx])
                kk.append(self.sample_keys[idx])

        return xx, kk

    def shuffle(self):
        xyz = list(zip(self.sample_xs, self.sample_ys, self.sample_keys))
        np.random.shuffle(xyz)
        self.sample_xs, self.sample_ys, self.sample_keys = zip(*xyz)
   
    def merge_samples(self, other_sampler):
        self.sample_xs.extend(other_sampler.sample_xs)
        self.sample_ys.extend(other_sampler.sample_ys)
        self.sample_keys.extend(other_sampler.sample_keys)