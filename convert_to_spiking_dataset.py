
import os
import sys
from tqdm import tqdm

import numpy as np
from scipy.sparse import coo_array

import torchvision

def poisson_spike_indexes(rate, tend, dt):
    if rate == 0:
        return np.array([])
    n_spikes = int(np.ceil(tend * rate))
    ISI = np.random.exponential(1/rate, n_spikes)
    spike_times = np.cumsum(ISI)
    idx = (spike_times / dt).astype(int)
    n_steps = int(tend / dt)
    idx = idx[idx < n_steps]
    return idx

def firing_rates_to_spike_trains(rates, tend, dt):
    row,col = [],[]
    for j,r in enumerate(rates):
        idx = np.unique(poisson_spike_indexes(r, tend, dt))
        for i in idx:
            row.append(i)
            col.append(j)
    data = np.ones(len(row))
    out = coo_array((data, (row,col)), shape=(int(tend / dt), len(rates)), dtype=np.int8)
    assert out.max() <= 1
    return out

if __name__ == '__main__':

    progname = os.path.basename(sys.argv[0])
    
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        print(f'usage: {progname} dataset-name')
        sys.exit(0)

    dataset_name = sys.argv[1]
    if dataset_name == 'mnist':
        dataset_fun = torchvision.datasets.MNIST
    elif dataset_name == 'fashion-mnist':
        dataset_fun = torchvision.datasets.FashionMNIST
    else:
        print(f"{progname}: supported dataset names are 'mnist' and 'fashion-mnist'.")
        sys.exit(1)

    root = os.path.join('notebooks', 'data', dataset_name)
    training_set = dataset_fun(root, train=True, transform=None, target_transform=None, download=True)
    test_set = dataset_fun(root, train=False, transform=None, target_transform=None, download=True)
    rows,cols = training_set.data.shape[1:]
    print('The examples are {}-by-{} pixels.'.format(rows, cols))

    X_train = (training_set.data.numpy() / 255.).reshape(len(training_set), -1)
    labels_train = training_set.targets.numpy()
    X_test = (test_set.data.numpy() / 255.).reshape(len(test_set), -1)
    labels_test  = test_set.targets.numpy()

    tend = 0.2
    dt = 1e-3
    max_firing_rate = 10.

    for X, labels, suffix in zip((X_train,X_test), (labels_train,labels_test), ('train','test')):
        print(f'Building {suffix} set...')
        spike_trains = []
        for i in tqdm(range(len(labels)), ascii=True, ncols=80):
            rates = X[i] * max_firing_rate
            spike_trains.append(firing_rates_to_spike_trains(rates, tend, dt))
        outfile = '{}-{}-spike-trains_tend={:.2f}_dt={:.3f}_max_rate={:.0f}.npz'.\
            format(dataset_name, suffix, tend, dt, max_firing_rate)
        np.savez_compressed(outfile, spike_trains=spike_trains)
        np.savez_compressed(f'{dataset_name}-{suffix}-labels.npz', labels=labels)

