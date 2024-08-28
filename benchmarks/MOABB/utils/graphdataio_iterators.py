import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric
import os
from utils.preparegraph import prepare_data

def find_sampled_node_indices(full_nodes, sampled_nodes):
    node_index = {node: idx for idx, node in enumerate(full_nodes)}
    sampled_indices = [node_index[node] for node in sampled_nodes if node in node_index]
    return sampled_indices

def remap_indices(edges):
    # Convert the input lists to a NumPy array for easier manipulation
    edges_array = np.array(edges)
    
    # Flatten the array to find all unique nodes
    unique_nodes = np.unique(edges_array)
    
    # Create a mapping from original nodes to new indices
    node_mapping = {node: index for index, node in enumerate(unique_nodes)}
    
    # Apply the mapping to remap the edges
    remapped_edges = np.vectorize(node_mapping.get)(edges_array)
    
    return remapped_edges


def get_edges(adj_matrix, channels, n_steps_channel_selection, seed_nodes=["Cz"]):
    """
    Returns the edge index tensor for the selected channels.

    Arguments
    ---------
    adj_matrix : np.ndarray
        Adjacency matrix representing connections between channels.
    channels : list of str
        List of all channel names.
    n_steps_channel_selection : int
        Number of steps used to select neighboring channels.
    seed_nodes : list of str, optional
        Seed channels from which neighboring channels are selected.

    Returns
    -------
    edge_index : torch.tensor
        Tensor containing edges between selected channels.
    selected_channels : list of str
        List of the selected channels.
    """
    # Sample the channels using the adjacency matrix
    selected_channels = get_neighbour_channels(
        adjacency_mtx=adj_matrix,
        ch_names=channels,
        n_steps=n_steps_channel_selection,
        seed_nodes=seed_nodes
    )

    # Get the indices of the selected channels
    sampled_indices = [channels.index(ch) for ch in selected_channels]

    # Create the subgraph adjacency matrix for the selected channels
    subgraph_adj_matrix = adj_matrix[np.ix_(sampled_indices, sampled_indices)]

    # Find the edges in the subgraph
    edge_indices = np.nonzero(subgraph_adj_matrix)
    edge_index = np.vstack((edge_indices[0], edge_indices[1]))

    # Re-map the edge indices to be within the range [0, len(selected_channels) - 1]
    remapped_edge_index = remap_indices(edge_index)

    return torch.tensor(remapped_edge_index, dtype=torch.long)





def get_idx_train_valid_classbalanced(idx_train, valid_ratio, y):
    idx_train = np.array(idx_train)
    nclasses = y[idx_train].max() + 1

    idx_valid = []
    for c in range(nclasses):
        to_select_c = idx_train[np.where(y[idx_train] == c)[0]]
        idx = np.linspace(0, to_select_c.shape[0] - 1, round(valid_ratio * to_select_c.shape[0]))
        idx = np.floor(idx).astype(int)
        tmp_idx_valid_c = to_select_c[idx]
        idx_valid.extend(tmp_idx_valid_c)

    idx_valid = np.array(idx_valid)
    idx_train = np.setdiff1d(idx_train, idx_valid)
    return idx_train, idx_valid

class GraphTensorDataset(Dataset):
    def __init__(self, features, labels, edge_index):
        self.features = features
        self.labels = labels
        self.edge_index = edge_index
        self.len = len(features)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return [self.features[idx],self.labels[idx], self.edge_index]

def create_dataset(xy, edge_index):
    x, y = xy
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = GraphTensorDataset(x_tensor, y_tensor, edge_index)
    return dataset

def get_dataloader(batch_size, xy_train, xy_valid, xy_test, edges):
    train_dataset = create_dataset(xy_train, edges)
    valid_dataset = create_dataset(xy_valid, edges)
    test_dataset = create_dataset(xy_test, edges)

    # Use PyTorch Geometric DataLoader
    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = torch_geometric.loader.DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

    return train_loader, valid_loader, test_loader

def crop_signals(x, srate, interval_in, interval_out):
    time = np.arange(interval_in[0], interval_in[1], 1 / srate)
    idx_start = np.argmin(np.abs(time - interval_out[0]))
    idx_stop = np.argmin(np.abs(time - interval_out[1]))
    return x[..., idx_start:idx_stop]

def get_neighbour_channels(adjacency_mtx, ch_names, n_steps=1, seed_nodes=["Cz"]):
    sel_channels = set(seed_nodes)
    for _ in range(n_steps):
        tmp_sel_channels = set()
        for node in seed_nodes:
            idx_node = np.where(node == np.array(ch_names))[0][0]
            idx_linked_nodes = np.where(adjacency_mtx[idx_node, :] > 0)[0]
            linked_channels = np.array(ch_names)[idx_linked_nodes]
            tmp_sel_channels.update(linked_channels)
        seed_nodes = tmp_sel_channels
        sel_channels.update(tmp_sel_channels)
    return list(sel_channels)

def sample_channels(x, adjacency_mtx, ch_names, n_steps, seed_nodes=["Cz"]):
    sel_channels = get_neighbour_channels(adjacency_mtx, ch_names, n_steps=n_steps, seed_nodes=seed_nodes)
    idx_sel_channels = [k for k, ch in enumerate(ch_names) if ch in sel_channels]

    if len(idx_sel_channels) != x.shape[1]:
        x = x[:, idx_sel_channels, :]
        sel_channels_ = np.array(ch_names)[idx_sel_channels]
        print(f"Sampling channels: {list(sel_channels_)}")
    else:
        print(f"Sampling all channels available: {ch_names}")
    return x

class GraphLeaveOneSessionOut:
    def __init__(self, seed):
        self.iterator_tag = "leave-one-session-out"
        np.random.seed(seed)

    def prepare(self, data_folder=None, cached_data_folder=None, dataset=None, batch_size=None, valid_ratio=None, target_subject_idx=None, target_session_idx=None, events_to_load=None, original_sample_rate=None, sample_rate=None, fmin=None, fmax=None, tmin=None, tmax=None, save_prepared_dataset=None, n_steps_channel_selection=None):
        interval = [tmin, tmax]

        data_dict = prepare_data(
            data_folder=data_folder,
            cached_data_folder=cached_data_folder,
            dataset=dataset,
            events_to_load=events_to_load,
            srate_in=original_sample_rate,
            srate_out=sample_rate,
            fmin=fmin,
            fmax=fmax,
            idx_subject_to_prepare=target_subject_idx,
            save_prepared_dataset=save_prepared_dataset,
        )

        x = data_dict["x"]
        y = data_dict["y"]
        srate = data_dict["srate"]
        original_interval = data_dict["interval"]
        metadata = data_dict["metadata"]

        if np.unique(metadata.session).shape[0] < 2:
            raise ValueError("The number of sessions in the dataset must be >= 2 for leave-one-session-out iterations")

        sessions = np.unique(metadata.session)
        sess_id_test = sessions[target_session_idx]
        sess_id_train = list(np.setdiff1d(sessions, [sess_id_test]))

        idx_train, idx_valid = [], []
        for s in sess_id_train:
            idx = np.where(metadata.session == s)[0]
            tmp_idx_train, tmp_idx_valid = get_idx_train_valid_classbalanced(idx, valid_ratio, y)
            idx_train.extend(tmp_idx_train)
            idx_valid.extend(tmp_idx_valid)
        idx_test = np.where(metadata.session == sess_id_test)[0]

        x_train = x[idx_train, ...]
        y_train = y[idx_train]
        x_valid = x[idx_valid, ...]
        y_valid = y[idx_valid]
        x_test = x[idx_test, ...]
        y_test = y[idx_test]

        if interval != original_interval:
            x_train = crop_signals(x=x_train, srate=srate, interval_in=original_interval, interval_out=interval)
            x_valid = crop_signals(x=x_valid, srate=srate, interval_in=original_interval, interval_out=interval)
            x_test = crop_signals(x=x_test, srate=srate, interval_in=original_interval, interval_out=interval)

        if n_steps_channel_selection is not None:
            x_train = sample_channels(x_train, data_dict["adjacency_mtx"], data_dict["channels"], n_steps=n_steps_channel_selection)
            x_valid = sample_channels(x_valid, data_dict["adjacency_mtx"], data_dict["channels"], n_steps=n_steps_channel_selection)
            x_test = sample_channels(x_test, data_dict["adjacency_mtx"], data_dict["channels"], n_steps=n_steps_channel_selection)

        x_train = np.swapaxes(x_train, -1, -2)
        x_valid = np.swapaxes(x_valid, -1, -2)
        x_test = np.swapaxes(x_test, -1, -2)

        sampled_edges = get_edges(data_dict["adjacency_mtx"], data_dict["channels"], n_steps_channel_selection=n_steps_channel_selection)

        train_loader, valid_loader, test_loader = get_dataloader(batch_size, (x_train, y_train), (x_valid, y_valid), (x_test, y_test), sampled_edges)

        datasets = {"train": train_loader, "valid": valid_loader, "test": test_loader}
        tail_path = os.path.join(self.iterator_tag, f"sub-{str(dataset.subject_list[target_subject_idx]).zfill(3)}", sessions[target_session_idx])
        return tail_path, datasets

class GraphLeaveOneSubjectOut:
    def __init__(self, seed):
        self.iterator_tag = "leave-one-subject-out"
        np.random.seed(seed)

    def prepare(self, data_folder=None, cached_data_folder=None, dataset=None, batch_size=None, valid_ratio=None, target_subject_idx=None, target_session_idx=None, events_to_load=None, original_sample_rate=None, sample_rate=None, fmin=None, fmax=None, tmin=None, tmax=None, save_prepared_dataset=None, n_steps_channel_selection=None):
        interval = [tmin, tmax]
        if len(dataset.subject_list) < 2:
            raise ValueError("The number of subjects in the dataset must be >= 2 for leave-one-subject-out iterations")

        data_dict = prepare_data(
            data_folder=data_folder,
            cached_data_folder=cached_data_folder,
            dataset=dataset,
            events_to_load=events_to_load,
            srate_in=original_sample_rate,
            srate_out=sample_rate,
            fmin=fmin,
            fmax=fmax,
            idx_subject_to_prepare=target_subject_idx,
            save_prepared_dataset=save_prepared_dataset,
        )

        x_test = data_dict["x"]
        y_test = data_dict["y"]
        original_interval = data_dict["interval"]
        srate = data_dict["srate"]

        subject_idx_train = [i for i in np.arange(len(dataset.subject_list)) if i != target_subject_idx]
        subject_ids_train = list(np.array(dataset.subject_list)[subject_idx_train])

        x_train, y_train, x_valid, y_valid = [], [], [], []
        for subject_idx in subject_idx_train:
            data_dict = prepare_data(
                data_folder=data_folder,
                cached_data_folder=cached_data_folder,
                dataset=dataset,
                events_to_load=events_to_load,
                srate_in=original_sample_rate,
                srate_out=sample_rate,
                fmin=fmin,
                fmax=fmax,
                idx_subject_to_prepare=subject_idx,
                save_prepared_dataset=save_prepared_dataset,
            )

            tmp_x_train = data_dict["x"]
            tmp_y_train = data_dict["y"]
            tmp_metadata = data_dict["metadata"]

            idx_train, idx_valid = [], []
            for session in np.unique(tmp_metadata.session):
                idx = np.where(tmp_metadata.session == session)[0]
                tmp_idx_train, tmp_idx_valid = get_idx_train_valid_classbalanced(idx, valid_ratio, tmp_y_train)
                idx_train.extend(tmp_idx_train)
                idx_valid.extend(tmp_idx_valid)

            tmp_x_valid = tmp_x_train[idx_valid, ...]
            tmp_y_valid = tmp_y_train[idx_valid]
            tmp_x_train = tmp_x_train[idx_train, ...]
            tmp_y_train = tmp_y_train[idx_train]

            x_train.extend(tmp_x_train)
            y_train.extend(tmp_y_train)
            x_valid.extend(tmp_x_valid)
            y_valid.extend(tmp_y_valid)

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        if interval != original_interval:
            x_train = crop_signals(x=x_train, srate=srate, interval_in=original_interval, interval_out=interval)
            x_valid = crop_signals(x=x_valid, srate=srate, interval_in=original_interval, interval_out=interval)
            x_test = crop_signals(x=x_test, srate=srate, interval_in=original_interval, interval_out=interval)

        if n_steps_channel_selection is not None:
            x_train = sample_channels(x_train, data_dict["adjacency_mtx"], data_dict["channels"], n_steps=n_steps_channel_selection)
            x_valid = sample_channels(x_valid, data_dict["adjacency_mtx"], data_dict["channels"], n_steps=n_steps_channel_selection)
            x_test = sample_channels(x_test, data_dict["adjacency_mtx"], data_dict["channels"], n_steps=n_steps_channel_selection)

        x_train = np.swapaxes(x_train, -1, -2)
        x_valid = np.swapaxes(x_valid, -1, -2)
        x_test = np.swapaxes(x_test, -1, -2)

        sampled_edges = get_edges(data_dict["adjacency_mtx"], data_dict["channels"],n_steps_channel_selection=n_steps_channel_selection)

        train_loader, valid_loader, test_loader = get_dataloader(batch_size, (x_train, y_train), (x_valid, y_valid), (x_test, y_test), sampled_edges)

        datasets = {"train": train_loader, "valid": valid_loader, "test": test_loader}
        tail_path = os.path.join(self.iterator_tag, f"sub-{str(dataset.subject_list[target_subject_idx]).zfill(3)}")
        return tail_path, datasets
