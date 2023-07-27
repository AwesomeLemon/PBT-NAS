# Extension of ReplayBufferStorage to use shared RAM. Episodes are stored there, and accessed by multiple agents
# Working with shared RAM from Python turned out to be finicky, see comments throughout for important points.
import gc
import os.path

import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset
from filelock import FileLock

import utils.general
from search.rl.shared_memory_fixed import SharedMemory, ShareableList
import multiprocessing

import utils

class ReplayBufferStorageSharedRam:
    def __init__(self, data_specs, replay_dir, unique_id):
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()
        self.episode_cnt = 0
        self.unique_id = unique_id


    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            # assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len, _ = fn.stem.split('_')
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        self._num_episodes += 1
        eps_len = episode_len(episode)
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}_{self.unique_id}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    lock = FileLock('/tmp/' + fn.name + '.lock')
    lock.acquire()
    try:
        with io.BytesIO() as bs:
            np.savez_compressed(bs, **episode)
            bs.seek(0)
            with fn.open('wb') as f:
                f.write(bs.read())
    finally:
        lock.release()

def store_np_array_in_shared_ram(np_array, name: str):
    shm = SharedMemory(name, create=True, size=np_array.nbytes)
    arr_in_shm = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
    arr_in_shm[:] = np_array
    arr_info_sl = ShareableList(list(np_array.shape) + [np_array.dtype.name], name=name + 'info')
    return arr_in_shm, arr_info_sl, shm

def get_np_array_from_shared_ram(name: str):
    sl = ShareableList(name=name + 'info')
    arr_info_sl = list(sl)
    shape = tuple(arr_info_sl[:-1])
    dtype_name = arr_info_sl[-1]
    shm = SharedMemory(name)
    arr_in_shm = np.ndarray(shape, dtype=np.dtype(dtype_name), buffer=shm.buf)
    return arr_in_shm, sl, shm

def store_episode_in_shared_ram(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}

    episode_shm = {}
    sl_refs = {}
    shm_refs = {}
    fn_name = fn.name

    for k in episode.keys():
        episode_shm[k], sl_refs[k], shm_refs[k] = store_np_array_in_shared_ram(episode[k], fn_name + k)

    return episode_shm, sl_refs, shm_refs

def load_from_shared_ram(fn):
    episode_shm = {}
    sl_refs = {}
    shm_refs = {}
    fn_name = fn.name

    for k in ['observation', 'action', 'reward', 'discount']:
        episode_shm[k], sl_refs[k], shm_refs[k] = get_np_array_from_shared_ram(fn_name + k)

    return episode_shm, sl_refs, shm_refs

def close_episode_wrap(episode_wrap, if_unlink=False):
    episode_shm, sl_refs, shm_refs = episode_wrap
    for k in ['observation', 'action', 'reward', 'discount']:
        del episode_shm[k]
        shm_refs[k].close()

        sl_refs[k].shm.close()

        if if_unlink:
            shm_refs[k].unlink()
            sl_refs[k].shm.unlink()
        del shm_refs[k]
        del sl_refs[k]
    # assert len(episode_shm) == 0
    del episode_shm

def episode_wrap_len(episode_wrap):
    return episode_wrap[0]['observation'].shape[0] - 1

def load_episode(fn, episode_wrap_dict, counter_dict):
    fn_name = fn.name
    # assure that no other process is trying to load the same episode
    lock = FileLock('/tmp/' + fn_name + '.lock')
    lock.acquire()
    try:
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)

        # check if the episode is already present in shared memory
        # but first check if we maybe already have access to it
        if fn_name + 'counter' not in counter_dict:
            counter_dict[fn_name + 'counter'] = None
            try:
                counter_dict[fn_name + 'counter'] = ShareableList(name=fn_name + 'counter')
            except:
                pass

        if counter_dict[fn_name + 'counter'] is None:
            counter_dict[fn_name + 'counter'] = ShareableList([1], name=fn_name + 'counter')

            episode_wrap_dict[fn] = store_episode_in_shared_ram(fn)
            # print(f'Shm: created {fn_name} (id={get_worker_id_or_none()})')
        else:
            if fn not in episode_wrap_dict:
                episode_wrap_dict[fn] = load_from_shared_ram(fn)

            counter_dict[fn_name + 'counter'][0] += 1

    finally:
        lock.release()


def get_worker_id_or_none():
    info = torch.utils.data.get_worker_info()
    if info is None:
        return None
    return info.id


def delete_episode_from_shm_and_disk(fn, episode_wrap_dict, counter_dict, delete_from_disk=True):
    '''
    Deletes shm references from this process; checks if any other process uses the file, and deletes it if not.
    Assumes that all the shm stuff exists (counter + [array+array_info for each array])
    Don't delete from disk if it's just clean-up after training step
    '''
    lock = FileLock('/tmp/' + fn.name + '.lock')
    lock.acquire()
    try:
        if_unlink = False
        if counter_dict[fn.name + 'counter'][0] == 1: # only this process
            sl_cnt = counter_dict.pop(fn.name + 'counter')
            if_unlink = True
            if delete_from_disk:
                # print(f'Unlinked {fn.name}')
                fn.unlink(missing_ok=True)
            shm_ref = sl_cnt.shm
            del sl_cnt
            shm_ref.close()
            shm_ref.unlink()
            del shm_ref
        else:
            counter_dict[fn.name + 'counter'][0] -= 1
            del counter_dict[fn.name + 'counter']

        ew = episode_wrap_dict.pop(fn)
        close_episode_wrap(ew, if_unlink)
        del ew

    finally:
        lock.release()

    try:
        if delete_from_disk and if_unlink:
            os.remove('/tmp/' + fn.name + '.lock')
    except:
        pass


class ReplayBufferSharedRam(IterableDataset):
    '''
    Note: I need to save refs_all, because once a variable referencing SharedMemory is garbage-collected,
          the underlying buffer is deleted, despite me not calling neither close() nor unlink().
          That is incredibly annoying, I'm glad someone figured this out before me
          https://stackoverflow.com/questions/63713241/segmentation-fault-using-python-shared-memory
          https://stackoverflow.com/questions/63106751/segfault-accessing-shared-memory-in-python
    '''
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        # self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self.n_sampled = 0
        self.paths_to_delete = []
        # self.refs_all = {'old': []}
        self.episode_wrap_dict = {}
        self.counter_dict = {}

        self._episode_fns_ignore = set()

        self.delete_delay_countdown_max = 10
        self.delete_delay_countdowns = {}

        self.shutdown = multiprocessing.Value('b', 0)

        self.shapes = None

    @staticmethod
    def destructor(_episode_fns, episode_wrap_dict, counter_dict, delete_delay_countdowns):
        '''
        Unfortunately, this is called in the main process, and not in dataloader worker processes, where
        self._episode_fns are actually stored.
        Workaround: have a shared memory value that is set to 1 when it's time to shut down;
            then, call next(iter) many times for all threads to shut down.
        '''
        while len(_episode_fns) > 0:
            eps_fn = _episode_fns.pop(0)
            delete_episode_from_shm_and_disk(eps_fn, episode_wrap_dict, counter_dict, delete_from_disk=False)

        for eps_fn in list(delete_delay_countdowns.keys()):
            try:
                delete_episode_from_shm_and_disk(eps_fn, episode_wrap_dict, counter_dict, delete_from_disk=False)
                del delete_delay_countdowns[eps_fn]
            except KeyError as e:
                print(traceback.format_exc())
                pass

        gc.collect()

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        # print(f'sampled {eps_fn.name}')
        # return self._episodes[eps_fn]
        return self.episode_wrap_dict[eps_fn][0]

    def _store_episode(self, eps_fn):
        try:
            load_episode(eps_fn, self.episode_wrap_dict, self.counter_dict)
        except:
            traceback.print_exc()
            return False

        eps_len = episode_wrap_len(self.episode_wrap_dict[eps_fn])

        while eps_len + self._size > self._max_size:
            # first, delete what's in the deletion queue:
            # (reason for this: if I delete shared memory while dataloader is constructing a batch (so, the control
            #   flow is outside this class & my code), it breaks.
            #   => delete with delay)
            popped = []
            for early_eps_fn_old in self.delete_delay_countdowns.keys():
                if self.delete_delay_countdowns[early_eps_fn_old] > 0:
                    self.delete_delay_countdowns[early_eps_fn_old] -= 1
                else:
                    popped.append(early_eps_fn_old) # can't modify while iterating
                    delete_episode_from_shm_and_disk(early_eps_fn_old, self.episode_wrap_dict, self.counter_dict)
            for p in popped:
                del self.delete_delay_countdowns[p]

            early_eps_fn = self._episode_fns.pop(0)
            self._size -= episode_wrap_len(self.episode_wrap_dict[early_eps_fn])

            self._episode_fns_ignore.add(early_eps_fn)
            self.delete_delay_countdowns[early_eps_fn] = self.delete_delay_countdown_max

        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._size += eps_len
        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _get_new_eps_fns(self):
        eps_fns = list(sorted(self._replay_dir.glob('*.npz')))
        return eps_fns

    def _preload_episodes(self):
        '''
        because I train networks in steps of X epochs, at every such step I need to load the episodes into buffer, and
            delete them from shared memory at the end (for careful memory management).
        '''
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0

        eps_fns = self._get_new_eps_fns()  # sorted(self._replay_dir.glob('*.npz'), reverse=True) #
        # load in random order so that different agents could load different ones in parallel
        np.random.shuffle(eps_fns)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len, _ = [int(x) for x in eps_fn.stem.split('_')[1:]]
            # print(f'{eps_idx=} {worker_id=}')
            if eps_idx % self._num_workers != worker_id:
                continue

            if eps_fn in self.episode_wrap_dict.keys():
                continue

            # added this check to not load just-deleted episodes.
            if eps_fn in self._episode_fns_ignore:
                continue

            if fetched_size + eps_len > self._max_size:
                break

            if not self._store_episode(eps_fn):
                # break
                continue  # replaced 'break' with 'continue' because why not?
            else:
                fetched_size += eps_len

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = self._get_new_eps_fns()
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len, _ = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue

            if eps_fn in self.episode_wrap_dict.keys():
                # break
                continue # Changed this to avoid the situation when new experiences come in faster than they can be used

            # added this check to not load just-deleted episodes.
            if eps_fn in self._episode_fns_ignore:
                continue

            if fetched_size + eps_len > self._max_size:
                break

            if not self._store_episode(eps_fn):
                continue
            else:
                fetched_size += eps_len
                break # added this: if you successfully added one episode, stop for now

    def _sample(self):
        try:
            self._try_fetch()
        except:
            print('Failed to fetch an episode')
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount

        if self.shapes is None:
            self.shapes = (obs.shape, action.shape, reward.shape, discount.shape, next_obs.shape)

        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        id = get_worker_id_or_none()
        # print(f"Started (id={id})")
        self._preload_episodes()
        if id is not None:
            worker_dataset_copy = torch.utils.data.get_worker_info().dataset
            sd = worker_dataset_copy.shutdown

        while True:
            if id is not None:
                if sd.value != 0:
                    ReplayBufferSharedRam.destructor(worker_dataset_copy._episode_fns,
                                                     worker_dataset_copy.episode_wrap_dict,
                                                     worker_dataset_copy.counter_dict,
                                                     worker_dataset_copy.delete_delay_countdowns
                                                     )
                    fake_data = []
                    for s in self.shapes:
                        fake_data.append(np.zeros(s))
                    fake_data = tuple(fake_data)
                    # finally:
                    #     lock.release()
                    yield fake_data
                else:
                    yield self._sample()
            else:
                yield self._sample()


def make_replay_loader_shared_ram(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    max_size_per_worker = max_size // max(1, num_workers)
    iterable = ReplayBufferSharedRam(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=125,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=utils.general.worker_init_fn)
    return loader