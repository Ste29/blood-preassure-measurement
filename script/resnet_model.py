import tensorflow as tf


def resnet_model():
    """ """

    filters = [64, 128, 256, 512]
    kernels = [3, 3, 3, 3]
    strides = [1, 1, 1, 1]

    # Inizio della rete
    sign = tf.keras.layers.Input(shape=[None, 1], dtype='float32')
    x = tf.keras.layers.Conv1D(64, 3, strides=1, padding='causal')(sign)
    # tf.keras.layers.Conv1D(filters=32, kernel_size=10, activation='relu', padding='same')

    for i in range(len(kernels)):
        x = _resnet_block(
            x,
            filters[i],
            kernels[i],
            strides[i])

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.AveragePooling1D(4, 1)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.CuDNNLSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.CuDNNLSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.CuDNNLSTM(128, return_sequences=True)(x)
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)

    model = tf.keras.Model(inputs=sign, outputs=x, name='resnet18')
    tf.keras.utils.plot_model(model=model, to_file='lstm_model.png',
                              show_shapes=True, show_layer_names=True)
    print(model.summary())
    return model


def _resnet_block(x, filters, kernel, stride):
    """Network block for ResNet."""
    # Struttura classica: batchnorm, relu, 2 strade una diretta e una che passa da
    # layer, batchnorm, relu, layer e a questo punto le 2 strade si sommano.
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    # print(x.shape[2])
    # Controllo sulle dimensioni perché altrimenti potrebbero non coincidere le
    # dimensioni quando vai a sommare la shortcut a ciò che è stato passato nel
    # layer (quando si aggiorna l'indice del ciclo)
    # Filters sono le feature maps! Per questo le guardi altrimenti quando cambia
    # rischi di trovarti un segnale con 64 che si somma a uno con 128
    if stride != 1 or filters != x.shape[2]:  # prima c'era x.shape[1]
        shortcut = _projection_shortcut(x, filters, stride)
    else:
        shortcut = x

    x = tf.keras.layers.Conv1D(filters, kernel, strides=stride, padding='causal')(x)
    #
    # x = tf.keras.layers.Dropout(0.5)(x)
    #
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv1D(filters, kernel, strides=1, padding='causal')(x)
    #
    # x = tf.keras.layers.Dropout(0.5)(x)
    #
    x = tf.keras.layers.add([x, shortcut])
    return x


def _projection_shortcut(x, out_filters, stride):
    # per poter far si che il segnale che prende la scorciatoia abbia la stessa
    # dimensione del segnale che esce dai 2 layer si prende il segnale originale e
    # lo si passa in un layer che ha lo stesso numero di feature maps che avrà
    # il segnale all'uscita dei 2 layer, kernel 1 solo perché non vuoi distorcerlo
    # e stride usi lo stride che usano i 2 layer, padding valid
    x = tf.keras.layers.Conv1D(out_filters, 1, strides=stride, padding='causal')(x)
    # filters, kernel size, strides
    return x
