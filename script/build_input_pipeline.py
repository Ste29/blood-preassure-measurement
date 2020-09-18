import tensorflow as tf


def build_input_pipeline(batch_size, valid_size):
    """Pipeline per caricare i dati sul grafo

    pz = indice del ciclo, è l'iesimo paziente che va usato come test
    """

    placeholder_X = tf.placeholder(tf.float32, [None, None, 1])
    placeholder_y = tf.placeholder(tf.float32, [None, None, 1])
    # Train dataset
    training_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X,
                                                           placeholder_y))

    training_dataset = training_dataset.shuffle(
        120000, reshuffle_each_iteration=True).repeat().batch(batch_size)
    training_dataset = training_dataset.prefetch(1)

    training_iterator = tf.data.make_initializable_iterator(training_dataset)

    # Test dataset
    heldout_dataset = tf.data.Dataset.from_tensor_slices((placeholder_X,
                                                          placeholder_y))

    heldout_dataset = heldout_dataset.batch(valid_size)
    heldout_dataset = heldout_dataset.prefetch(1)

    heldout_iterator = tf.data.make_initializable_iterator(heldout_dataset)

    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    feedable_iterator = tf.compat.v1.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)

    signal, BP = feedable_iterator.get_next()

    return signal, BP, handle, training_iterator, heldout_iterator, \
           placeholder_X, placeholder_y