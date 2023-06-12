import math
import torch
import os
import argparse
import numpy as np
import itertools
from tqdm import tqdm
from utils import load_model, move_to
from utils.data_utils import save_dataset
import csv

from torch.utils.data import DataLoader
import time
from datetime import timedelta
import pickle
from utils.functions import parse_softmax_temperature
import matplotlib.pyplot as plt
mp = torch.multiprocessing.get_context('spawn')


def get_best(sequences, cost, ids=None, batch_size=None):
    """
    Ids contains [0, 0, 0, 1, 1, 2, ..., n, n, n] if 3 solutions found for 0th instance, 2 for 1st, etc
    :param sequences:
    :param lengths:
    :param ids:
    :return: list with n sequences and list with n lengths of solutions
    """
    if ids is None:
        idx = cost.argmin()
        return sequences[idx:idx+1, ...], cost[idx:idx+1, ...]

    splits = np.hstack([0, np.where(ids[:-1] != ids[1:])[0] + 1])
    mincosts = np.minimum.reduceat(cost, splits)

    group_lengths = np.diff(np.hstack([splits, len(ids)]))
    all_argmin = np.flatnonzero(np.repeat(mincosts, group_lengths) == cost)
    result = np.full(len(group_lengths) if batch_size is None else batch_size, -1, dtype=int)

    result[ids[all_argmin[::-1]]] = all_argmin[::-1]

    return [sequences[i] if i >= 0 else None for i in result], [cost[i] if i >= 0 else math.inf for i in result]


def eval_dataset_mp(args):
    (dataset_path, width, softmax_temp, opts, i, num_processes) = args

    model, _ = load_model(opts.model)
    val_size = opts.val_size // num_processes
    dataset = model.problem.make_dataset(filename=dataset_path, num_samples=val_size, offset=opts.offset + val_size * i)
    device = torch.device("cuda:{}".format(i))

    return _eval_dataset(model, dataset, width, softmax_temp, opts, device)

def get_tasks(dataset, task_size):
    for i in range(dataset.size):
        data = dataset.data[i]
        loc = data['loc'][0:task_size]
        deadline = data['deadline'][0:task_size]
        dataset.data[i]['loc'] = loc
        dataset.data[i]['deadline'] = deadline

    return dataset


def eval_dataset(dataset_path, width, softmax_temp, opts):
    # Even with multiprocessing, we load the model here since it contains the name where to write results
    model, _ = load_model(opts.model)
    use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.multiprocessing:
        assert use_cuda, "Can only do multiprocessing with cuda"
        num_processes = torch.cuda.device_count()
        assert opts.val_size % num_processes == 0

        with mp.Pool(num_processes) as pool:
            results = list(itertools.chain.from_iterable(pool.map(
                eval_dataset_mp,
                [(dataset_path, width, softmax_temp, opts, i, num_processes) for i in range(num_processes)]
            )))

    else:
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dataset = model.problem.make_dataset(filename=dataset_path, num_samples=opts.val_size, offset=opts.offset)
        # dataset = get_tasks(dataset, task_size)
        results = _eval_dataset(model, dataset, width, softmax_temp, opts, device)

    # This is parallelism, even if we use multiprocessing (we report as if we did not use multiprocessing, e.g. 1 GPU)
    parallelism = opts.eval_batch_size

    costs, task, durations, tours, _ = zip(*results)  # Not really costs since they should be negative


    dataset_basename, ext = os.path.splitext(os.path.split(dataset_path)[-1])
    model_name = "_".join(os.path.normpath(os.path.splitext(opts.model)[0]).split(os.sep)[-2:])
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, model.problem.NAME, dataset_basename)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}-{}{}-t{}-{}-{}{}".format(
            dataset_basename, model_name,
            opts.decode_strategy,
            width if opts.decode_strategy != 'greedy' else '',
            softmax_temp, opts.offset, opts.offset + len(costs), ext
        ))

    return results, dataset


def _eval_dataset(model, dataset, width, softmax_temp, opts, device):

    model.to(device)
    model.eval()

    model.set_decode_type(
        "greedy" if opts.decode_strategy in ('bs', 'greedy') else "sampling",
        temp=softmax_temp)

    dataloader = DataLoader(dataset, batch_size=opts.eval_batch_size)

    results = []
    tasks_done_total = []
    for batch in tqdm(dataloader, disable=opts.no_progress_bar):
        batch = move_to(batch, device)

        start = time.time()
        with torch.no_grad():
            if opts.decode_strategy in ('sample', 'greedy'):
                if opts.decode_strategy == 'greedy':
                    assert width == 0, "Do not set width when using greedy"
                    assert opts.eval_batch_size <= opts.max_calc_batch_size, \
                        "eval_batch_size should be smaller than calc batch size"
                    batch_rep = 1
                    iter_rep = 1
                elif width * opts.eval_batch_size > opts.max_calc_batch_size:
                    assert opts.eval_batch_size == 1
                    assert width % opts.max_calc_batch_size == 0
                    batch_rep = opts.max_calc_batch_size
                    iter_rep = width // opts.max_calc_batch_size
                else:
                    batch_rep = width
                    iter_rep = 1
                assert batch_rep > 0
                # This returns (batch_size, iter_rep shape)
                sequences, costs, tasks_done, robot_seq = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
                tasks_done_total.extend(tasks_done)
                batch_size = len(costs)
                ids = torch.arange(batch_size, dtype=torch.int64, device=costs.device)
            else:
                assert opts.decode_strategy == 'bs'

                cum_log_p, sequences, costs, ids, batch_size = model.beam_search(
                    batch, beam_size=width,
                    compress_mask=opts.compress_mask,
                    max_calc_batch_size=opts.max_calc_batch_size
                )

        if sequences is None:
            sequences = [None] * batch_size
            costs = [math.inf] * batch_size
        else:
            sequences, costs = get_best(
                sequences.cpu().numpy(), costs.cpu().numpy(),
                ids.cpu().numpy() if ids is not None else None,
                batch_size
            )
            robot_seq = [r_s.cpu().numpy() for r_s in robot_seq]
        duration = time.time() - start
        i = 0
        for seq, cost, r_q in zip(sequences, costs, robot_seq):
            if model.problem.NAME == "tsp":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME == "mrta":
                seq = seq.tolist()  # No need to trim as all are same length
            elif model.problem.NAME in ("cvrp"):
                seq = np.trim_zeros(seq).tolist() + [0]  # Add depot
            else:
                assert False, "Unkown problem: {}".format(model.problem.NAME)
            # Note VRP only
            results.append({"cost":cost, "tasks_done": tasks_done_total[i][0],"total_duration":duration, "sequence":seq, "robot_seq":r_q.tolist()})
            i +=1
    # plot tasks done here
    # plt.plot(tasks_done_total)
    # plt.show()
    return results

def ind2coord(seq, dataset, size=8):
    seq = np.array(seq)
    locs = dataset.data[0]["loc"].numpy()
    locs[:,:3] = locs[:,:3]*size
    trajs = []
    for uav in range(seq.shape[1]):
        tmp = []
        u, ind = np.unique(seq[:, uav], return_index=True)
        for idx in u[np.argsort(ind)]:
            tmp.append(locs[idx-1].tolist())
        trajs.append(tmp)

    return trajs

if __name__ == "__main__":
    model_file = "/home/rsl/ysl/attention-learn-to-route/outputs/mrta_162/run_20230612T132750/epoch-99.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', default=["data/mrta/50_nodes_mrta.pkl"], help="Filename of the dataset(s) to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--val_size', type=int, default=100,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--offset', type=int, default=0,
                        help='Offset where to start in dataset (default 0)')
    parser.add_argument('--eval_batch_size', type=int, default=10,
                        help="Batch size to use during (baseline) evaluation")
    # parser.add_argument('--decode_type', type=str, default='greedy',
    #                     help='Decode type, greedy or sampling')
    parser.add_argument('--width', type=int, nargs='+',
                        help='Sizes of beam to use for beam search (or number of samples for sampling), '
                             '0 to disable (default), -1 for infinite')
    parser.add_argument('--decode_strategy', default="greedy", type=str,
                        help='Beam search (bs), Sampling (sample) or Greedy (greedy)')
    parser.add_argument('--softmax_temperature', type=parse_softmax_temperature, default=1,
                        help="Softmax temperature (sampling or bs)")
    parser.add_argument('--model', default=model_file, type=str)
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--compress_mask', action='store_true', help='Compress mask into long')
    parser.add_argument('--max_calc_batch_size', type=int, default=10000, help='Size for subbatches')
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Use multiprocessing to parallelize over multiple GPUs')

    opts = parser.parse_args()

    assert opts.o is None or (len(opts.datasets) == 1 and len(opts.width) <= 1), \
        "Cannot specify result filename with more than one dataset or more than one width"

    widths = opts.width if opts.width is not None else [0]
    # all_files = "data/mrta_taptc/taptc_data_sets_n_100.pkl"
    # all_files = "data/mrta_gini/taptc_data_sets_n_100.pkl"
    # file_n = open(all_files, 'rb')
    # datasets = pickle.load(file_n)
    tot = []
    for width in widths:
        dataset_path = "./data/mrta/162_nodes_+mrta.pkl" # manual
        results, dataset = eval_dataset(dataset_path, width, opts.softmax_temperature, opts)
        traj = ind2coord(results[0]["robot_seq"], dataset, size=8)
        # tot.append(results[0]['tasks_done'])

        with open('robot_route.pickle', 'wb') as handle:
            pickle.dump(traj, handle)

        break


    


# with open('CapAM.csv', 'w') as f:
#     write = csv.writer(f)
#     write.writerows((np.array(tot).T).reshape((96,1)).tolist())