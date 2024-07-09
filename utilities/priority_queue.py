# imports
import numpy as np
import random



# SumTree
class SumTree:
    write = 0
    def __init__(self, capacity):
        # maximum elements the tree can hold
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # an array holding the actual experience data
        self.data = np.zeros(capacity, dtype=object)
        # current number of elements in the data array
        self.n_entries = 0

    def _propagate(self, idx, change):
        """
        recursively updates the tree nodes from a given index to the root, with a 'change' value
        """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        recursively finds and returns the index of a leaf node with a give value s
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        # returns the value of the root node (sum of all priorities)
        return self.tree[0]


    def add(self, p, data):
        # adds a new priority and corresponding data sample to the tree, updates relevant nodes
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        # updates priority at given index and propogates the change up
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        # retrieve priority and data sample of a given value s
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PER_Buffer:  # stored as ( s, a, r, s_ ) in SumTree
    """
    prioritizie experience replay buffer class - utilizies 'sumtree' data structure to
    store and sample experiences based on priorities.
    """
    # small constant ensures that priorities are never zero
    e = 0.01
    # degree to which priorities are used (0 - uniform sampling)
    a = 0.8
    # initial value for importance sampling weights to correct the bias introduced by prioritizied sampling
    beta = 0.4
    # the amount by which 'beta' is incremented after each sampling (similiar to 'gamma')
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        # instance of treesum
        self.tree = SumTree(capacity)
        # maximum number of elements in buffer
        self.capacity = capacity

    def _get_priority(self, error):
        """
        calculates the priority of a sample based on it's error
        a - determines how much prioriziation is applied:
         alpha tradeoff: (uniform / fully priorizied)
         alpha = 0 uniform samplign
         alpha = 1 fully priorizied
         returns the priority of a sample by taking an absoulute given error. this priority value then used in the
         sumTree to determine the likelihood of a sample beign chosen during the sampling process.
        """

        return (np.abs(error.detach().cpu().numpy()) + self.e) ** self.a

    def add(self, error, sample):
        """
        adds a new experience to the buffer with a priority derived from the given error.
        """
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        """
        samples n experiences from buffer based on priorities.
        returns batch of samples, their indices in the 'SumTree' and their importance-sampling weights.
        """
        # list to store the sampled experiences
        batch = []
        # list to store the indices of the sampled experiences in the tree
        idxs = []
        # segment size for diving the total priority range into 'n' segment
        segment = self.tree.total() / n
        # list to store the priorizies of the sampled experiences
        priorities = []
        # increment beta until 1 - to compute importance-sampling weights - does not exceed 1
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        # sample n experiences
        for i in range(n):
            # define the range [a,b] for each segment (to sample from)
            a = segment * i
            b = segment * (i + 1)
            # randomly select a 's' from the segmenet [a,b] - determines which experience to sample
            # s locates a specific leaf node (experience, sample) in the tree whose cumulative priority is closest to 's'
            s = random.uniform(a, b)
            # retrive index, priority and data from the tree
            # the data, compared to s, is the actual experience (state, action, reward) stored in the buffer.
            # s determines location in the tree, the corresponding data at the location is retrieved.
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        """
        returns:
        batches: batch of samples
        indxs: the indices of the samples in the SumTree
        is_weight: the importance-sampling weights for the samples.
        """
        return batch, idxs, is_weight

    def update(self, idx, error):
        """
        updates the priority of an expereince at a given index based on the new error value
        """
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def get_buffer_len(self):
        return self.tree.n_entries
