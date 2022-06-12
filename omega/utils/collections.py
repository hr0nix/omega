import abc

import numpy as np


def get_dict_slice(d, keys):
    result = {}
    for key in keys:
        if key in d:
            result[key] = d[key]
    return result


class PrioritizedSamplerBase(abc.ABC):
    @abc.abstractmethod
    def __len__(self):
        pass

    @abc.abstractmethod
    def add(self, item, priority):
        pass

    @abc.abstractmethod
    def update_priority(self, item, new_priority):
        pass

    @abc.abstractmethod
    def get_priority(self, item):
        pass

    @abc.abstractmethod
    def remove(self, item):
        pass

    @abc.abstractmethod
    def sample(self, num_items):
        pass


class LinearPrioritizedSampler(PrioritizedSamplerBase):
    def __init__(self, max_items, alpha=1.0, epsilon=1e-3):
        self._alpha = alpha
        self._epsilon = epsilon

        self._priorities = np.zeros(max_items, dtype=np.float32)
        self._index_to_item = []
        self._item_to_index = {}

    def __len__(self):
        return len(self._item_to_index)

    def add(self, item, priority):
        new_index = len(self)

        if new_index == self._priorities.shape[0]:
            raise RuntimeError('Maximum sampler capacity exceeded!')
        if item in self._item_to_index:
            raise ValueError('The item has already been added, use update_priority() instead.')
        if priority <= 0:
            raise ValueError(f'Priorities must be positive, got {priority}')

        self._item_to_index[item] = new_index
        self._index_to_item.append(item)
        self._priorities[new_index] = priority

    def get_priority(self, item):
        index = self._item_to_index.get(item)

        if index is None:
            raise ValueError('The item has not been added before.')

        return self._priorities[index]

    def update_priority(self, item, new_priority):
        index = self._item_to_index.get(item)

        if index is None:
            raise ValueError('The item has not been added before, use add() instead.')
        if new_priority <= 0:
            raise ValueError(f'Priorities must be positive, got {new_priority}')

        self._priorities[index] = new_priority

    def remove(self, item):
        index = self._item_to_index.get(item)

        if index is None:
            raise ValueError('The item has not been added before.')

        del self._item_to_index[item]
        last_item = self._index_to_item.pop()

        if last_item != item:
            # Move the last item into the position that has been freed
            self._item_to_index[last_item] = index
            self._index_to_item[index] = last_item
            self._priorities[index] = self._priorities[len(self)]

    def sample(self, num_items):
        num_candidates = len(self)
        if num_candidates == 0:
            raise RuntimeError('There are no items to sample.')

        probs = self._priorities[:num_candidates] + self._epsilon
        probs = np.power(probs, self._alpha)
        probs /= np.sum(probs)

        chosen_indices = np.random.choice(a=num_candidates, size=num_items, p=probs)
        return [
            self._index_to_item[index]
            for index in chosen_indices
        ]
