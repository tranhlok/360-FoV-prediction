import torch.nn.utils.rnn as rnn_utils

def collate_fn(batch):
    features, gaze_direction = zip(*batch)

    # Pad sequences to the length of the longest sequence in the batch
    padded_features = rnn_utils.pad_sequence(features, batch_first=True, padding_value=0)
    padded_gaze_direction = rnn_utils.pad_sequence(gaze_direction, batch_first=True, padding_value=0)

    return padded_features, padded_gaze_direction