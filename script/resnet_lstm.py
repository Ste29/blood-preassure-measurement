import os
from datetime import timedelta, datetime
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
import scipy.io
import matplotlib.pyplot as plt
from build_input_pipeline import *
from preprocess import *
from resnet_model import *


# ################################ Params ##############################################################################
learning_rate = default = 0.0001  # Initial learning rate
epochs = 5  # Number of epochs to train for
batch_size = 32 # 1300 -100 epoche, 256 - 20
# eval_freq = 10*len(x_train)/batch_size  # validate the model every 10 epochs
valid_size = 100

Normalize_input = 1
Normalize_output = 1
Normalize_5s = 0
Puntuale = 0
Quantizzazione = 0

divisore = 250  # 625  # 250
# SIGN_SHAPE = [divisore, 1]
iterazione = 0

# ################################ Load Data ###########################################################################
path_train = "/content/gdrive/My Drive/Tesi/Dataset/DS_RIDOTTI/ds con limiti su abp/finalds.mat"
path_valid = "/content/gdrive/My Drive/Tesi/Dataset/DS_RIDOTTI/ds con limiti su abp/finalval.mat"
data = scipy.io.loadmat(path_train)["final_ds"]

indici = np.unique(data[:,75000])

train, test = split_data(data, indici, iterazione)
nSamp_train = train.shape
nSamp_test = test.shape

train, test, min_output, max_output = preprocess(train, test, divisore, Quantizzazione,
                    Normalize_input, Normalize_output, Normalize_5s, Puntuale)

# ################################ Define graph ########################################################################
# Build Input pipeline
with tf.name_scope("Dataset"):
    (segnale, classe, handle, training_iterator, heldout_iterator,
     X, y) = build_input_pipeline(batch_size, valid_size)

# Build the network
# Tu vuoi il segnale in formato: Batch, lunghezza, canali (tipo RGB, qua invece
# puoi usare le derivate del segnale)
with tf.name_scope("CNN"):
    model = resnet_model()

logits = model(segnale)

with tf.name_scope("loss"):
    loss = tf.losses.huber_loss(classe, logits, delta=0.003)
    # loss = tf.losses.mean_squared_error(classe, logits)

with tf.name_scope("train"):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    if Normalize_output == 1:
        classe = tf.add(tf.math.multiply(classe, (max_output - min_output)), min_output)
        logits = tf.add(tf.math.multiply(logits, (max_output - min_output)), min_output)

    MAE_train, MAE_train_update = tf.metrics.mean_absolute_error(classe, logits)
    MSE_train, MSE_train_update = tf.metrics.mean_squared_error(classe, logits)

with tf.name_scope("valid"):
    MAE_valid, MAE_valid_update = tf.metrics.mean_absolute_error(classe, logits)
    MSE_valid, MSE_valid_update = tf.metrics.mean_squared_error(classe, logits)

loss_summary = tf.summary.scalar("loss", loss)
RMSE_train_summary = tf.summary.scalar("RMSE_train", tf.math.sqrt(MSE_train))
RMSE_valid_summary = tf.summary.scalar("RMSE_valid", tf.math.sqrt(MSE_valid))
MAE_train_summary = tf.summary.scalar("MAE_train", MAE_train)
MAE_valid_summary = tf.summary.scalar("MAE_valid", MAE_valid)

# init_op = tf.global_variables_initializer()
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

stream_vars_train = [
    v for v in tf.local_variables() if "train/" in v.name]
reset_train_op = tf.compat.v1.variables_initializer(stream_vars_train)

stream_vars_valid = [
    v for v in tf.local_variables() if "valid/" in v.name]
reset_valid_op = tf.compat.v1.variables_initializer(stream_vars_valid)

saver = tf.train.Saver()  # Nodo di salvataggio

# ################################ Define graph ########################################################################

# for pz in indici:
for pz in range(1):
    # paz = int(pz)
    paz = 11  # ##############################################  ricorda di sistemare tutto se lo fai per paziente
    print(f"Testing on {paz}")

    # Creazione cartella per conservare i dati di tensorboard
    now = time.strftime("%Y_%m_%d-%H_%M_%S")
    root_logdir = f"/content/gdrive/My Drive/Tesi/Dataset/tf_logs/{paz}/"
    logdir = "{}/run-{}/".format(root_logdir, now)
    root_savedir = f"/content/gdrive/My Drive/Tesi/Dataset/saves/{paz}/"

    file_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())

    training_steps = int(round(epochs * (train.shape[0] / batch_size)))
    print(f"training_steps: {training_steps}, test on paz: {pz}, len trainset: {train.shape[0]}")
    # print(f"training_steps: {training_steps}, len trainset: {data.shape[0]}")

    count = 1

    with tf.Session() as sess:
        t1 = time.time()
        sess.run(init_op)

        # Run the training loop

        train_handle = sess.run(training_iterator.string_handle())
        heldout_handle = sess.run(heldout_iterator.string_handle())

        sess.run(training_iterator.initializer,
                 feed_dict={handle: train_handle,
                            X: train[:, 0:divisore],
                            y: train[:, divisore:]})

        for step in range(training_steps):
            epoca = int(round(((step * batch_size) / train.shape[0])))

            _ = sess.run([train_op, MAE_train_update, MSE_train_update],
                         feed_dict={handle: train_handle})

            # training eval
            if step % 500 == 0:  # (len(x_train)//batch_size) == 0:

                loss_value, MSE_t, MAE_t, summary_loss, \
                sRMSE_train, sMAE_train = sess.run([loss, MSE_train,
                                                    MAE_train, loss_summary, RMSE_train_summary,
                                                    MAE_train_summary], feed_dict={handle: train_handle})

                t2 = time.time()
                delta = t2 - t1
                print(
                    "Step: {:>3d} Loss: {:.3f} MSE: {:.3f} MAE: {:.3f} time: {:.2f}".format(
                        step, loss_value, np.sqrt(MSE_t), MAE_t, delta))
                t1 = time.time()

                file_writer.add_summary(summary_loss, step)
                file_writer.add_summary(sRMSE_train, step)
                file_writer.add_summary(sMAE_train, step)
                sess.run(reset_train_op)

            # save
            # if (step + 1) % 2000 == 0: #(training_steps/4) == 0:
            if (step + 1) % int(training_steps / 4) == 0:
                savedir = "{}/run-{}/{}/".format(root_savedir, now, epoca)
                salvataggio = savedir + "LSTM.ckpt"

                # Creo la cartella per poter salvare
                if tf.io.gfile.exists(savedir):
                    tf.compat.v1.logging.warning(
                        "Warning: deleting old log directory at {}".format(savedir))
                    tf.io.gfile.rmtree(savedir)

                tf.io.gfile.makedirs(savedir)

                save_path = saver.save(sess, salvataggio)

                # test eval
            if (step + 1) % 1000 == 1:  # eval_freq == 0:

                sess.run(heldout_iterator.initializer,
                         feed_dict={handle: heldout_handle,
                                    X: test[:, 0:divisore],
                                    y: test[:, divisore:]})

                try:
                    while True:
                        sess.run([MAE_valid_update, MSE_valid_update],
                                 feed_dict={handle: heldout_handle})

                except tf.errors.OutOfRangeError:
                    pass

                MSE_v, MAE_v, sRMSE_valid, \
                sMAE_valid = sess.run([MSE_valid, MAE_valid, \
                                       RMSE_valid_summary, MAE_valid_summary],
                                      feed_dict={handle: heldout_handle})

                print("\t ... Validation MSE: {:.3f} MAE: {:.3f}".format(np.sqrt(MSE_v), MAE_v))
                file_writer.add_summary(sRMSE_valid, step)
                file_writer.add_summary(sMAE_valid, step)
                sess.run(reset_valid_op)

        file_writer.close()

# ################################ testing #############################################################################
with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, f"{salvataggio}/LSTM.ckpt")
    # print(1)

    train_handle = sess.run(training_iterator.string_handle())
    heldout_handle = sess.run(heldout_iterator.string_handle())

    sess.run(heldout_iterator.initializer,
             feed_dict={handle: heldout_handle,
                        X: val[:, 0:divisore],
                        y: val[:, divisore:]})

    grezzi = []

    try:
        while True:
            parziali, _, _ = sess.run([logits, MAE_valid_update, MSE_valid_update],
                                      feed_dict={handle: heldout_handle})
            grezzi.append(parziali)

    except tf.errors.OutOfRangeError:
        pass

    MSE_v, MAE_v = sess.run([MSE_valid, MAE_valid],
                            feed_dict={handle: heldout_handle})

    print("\t ... Validation MSE: {:.3f} MAE: {:.3f}".format(np.sqrt(MSE_v), MAE_v))

scipy.io.savemat("dati_ricostruiti.mat", mdict={'data': grezzi})
scipy.io.savemat("norm_output.mat", mdict={'max_output': max_output, "min_output": min_output})
