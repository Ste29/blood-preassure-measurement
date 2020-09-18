import numpy as np


def preprocess(data, val, divisore, Quantizzazione, Normalize_input, Normalize_output, Normalize_5s, Puntuale):
    if Quantizzazione == 1:
        quantizing_bits = 4
        quantizing_levels = 2 ** quantizing_bits / 2
        quantizing_step = 1. / quantizing_levels
        val[:, 75001:150001] = np.round(val[:, 75001:150001] / quantizing_step) * quantizing_step
        data[:, 75001:150001] = np.round(data[:, 75001:150001] / quantizing_step) * quantizing_step

    if Normalize_input == 1:
        # Standardization
        val[:, 0:75000] = (val[:, 0:75000] - np.mean(data[:, 0:75000])) / (np.std(data[:, 0:75000]))
        data[:, 0:75000] = (data[:, 0:75000] - np.mean(data[:, 0:75000])) / (np.std(data[:, 0:75000]))
        # Normalization
        # min_input = np.min(data[:,0:75000])
        # max_output = np.max(data[:,0:75000])
        # val[:,0:75000] = ( val[:,0:75000] - np.min(data[:,0:75000]) ) / ( np.max(data[:,0:75000]) - np.min(data[:,0:75000]) )
        # data[:,0:75000] = ( data[:,0:75000] - np.min(data[:,0:75000]) ) / ( np.max(data[:,0:75000]) - np.min(data[:,0:75000]) )

    if Normalize_output == 1:
        min_output = np.min(data[:, 75001:150001])
        max_output = np.max(data[:, 75001:150001])
        val[:, 75001:150001] = (val[:, 75001:150001] - np.min(data[:, 75001:150001])) / (
                    np.max(data[:, 75001:150001]) - np.min(data[:, 75001:150001]))
        data[:, 75001:150001] = (data[:, 75001:150001] - np.min(data[:, 75001:150001])) / (
                    np.max(data[:, 75001:150001]) - np.min(data[:, 75001:150001]))

    if Normalize_5s == 1:
        min_output = np.min(data[:, 150001:])
        max_output = np.max(data[:, 150001:])
        val[:, 150001:] = (val[:, 150001:] - np.min(data[:, 150001:])) / (
                    np.max(data[:, 150001:]) - np.min(data[:, 150001:]))
        data[:, 150001:] = (data[:, 150001:] - np.min(data[:, 150001:])) / (
                    np.max(data[:, 150001:]) - np.min(data[:, 150001:]))

    if Puntuale == 1:

        data = np.concatenate(
            (np.reshape(
                data[:, :75000], [np.int((data.shape[0] * data[:, 0:75000].shape[1]) / 625), 625]),
             np.reshape(
                 data[:, 150001:], [np.int((data.shape[0] * data[:, 150001:].shape[1]) / 2), 2])),
            axis=1)

        val = np.concatenate(
            (np.reshape(
                val[:, :75000], [np.int((val.shape[0] * val[:, 0:75000].shape[1]) / 625), 625]),
             np.reshape(
                 val[:, 150001:], [np.int((val.shape[0] * val[:, 150001:].shape[1]) / 2), 2])),
            axis=1)

        data = np.expand_dims(data, axis=2)
        val = np.expand_dims(val, axis=2)

    elif Puntuale == 0:
        data = np.concatenate(
            (np.reshape(
                data[:, :75000], [np.int((data.shape[0] * data[:, 0:75000].shape[1]) / divisore), divisore]),
             np.reshape(
                 data[:, 75001:150001],
                 [np.int((data.shape[0] * data[:, 75001:150001].shape[1]) / divisore), divisore])),
            axis=1)

        val = np.concatenate(
            (np.reshape(
                val[:, :75000], [np.int((val.shape[0] * val[:, 0:75000].shape[1]) / divisore), divisore]),
             np.reshape(
                 val[:, 75001:150001], [np.int((val.shape[0] * val[:, 75001:150001].shape[1]) / divisore), divisore])),
            axis=1)

        data = np.expand_dims(data, axis=2)
        val = np.expand_dims(val, axis=2)

    return data, val, min_output, max_output


def split_data(data, indici, iterazione):
    test = np.array([np.where(data[:, 75000] == i)[0]
                     for i in indici[iterazione * 5:(iterazione + 1) * 5]])
    test = np.reshape(test, 5 * 18)

    train = [i not in test for i in range(data.shape[0])]

    return data[train, :], data[test, :]
