import os
import pickle
import random
import sys
from collections import Counter

import geopy
import joblib
from scipy import spatial
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from byt5_model import ByT5ClusteredClassifierDataset, ByT5_classifier
import torch
from utils import distance, get_dist, get_evaluation_error_predict, read_scaler_and_vocab, read_csv_data, \
    read_train_test_data, add_cluster_id_column, add_subcluster_id_column, get_distances, distance_from_pred, \
    true_distance_from_pred, true_distance_from_pred_cluster, prepare_subclusters
import pandas as pd
import sklearn
import math
import sklearn.model_selection
from tqdm import tqdm
from torch import nn
import numpy as np
import argparse
import logging
import logging.handlers
from sklearn.neighbors import BallTree
import wandb
from pathlib import Path


def validate_rel(epoch, model, valid_generator, logger, show_progress=False, no_save=False, cluster_id=None, arch=None):
    model.eval()
    loss_ls = []
    texts = []
    distance_ls = []
    true_distance_ls = []
    true_clusters = []
    pred_clusters = []
    true_lats = []
    true_lngs = []
    if show_progress:
        generator = tqdm(valid_generator)
    else:
        generator = valid_generator
    for batch in generator:
        te_feature, te_label_true, text, te_lat, te_lng, te_langs = batch
        te_feature = te_feature.to(device)
        te_label_true = te_label_true.to(device)
        te_langs = te_langs.to(device)
        with torch.no_grad():
            te_predictions = model(te_feature, te_langs)
        te_loss = custom_loss(te_predictions, te_label_true).mean()
        texts.append(text)
        loss_ls.append(te_loss.item())
        true_lats.append(te_lat.detach().cpu())
        true_lngs.append(te_lng.detach().cpu())
        true_clusters.append(te_label_true.detach().cpu())
        pred_clusters.append(te_predictions.detach().cpu())
        true_distance_ls.append(
            true_distance_from_pred(te_predictions.detach().cpu(), te_lat.detach().cpu(), te_lng.detach().cpu(),
                                    cluster_df))
        loss_ls.append(te_loss.item())
    te_loss = sum(loss_ls) / len(loss_ls)
    true_distance_ls = torch.cat(true_distance_ls, 0)
    true_distance_ls = pd.Series(true_distance_ls.numpy())
    true_lats = torch.cat(true_lats, 0).numpy()
    true_lngs = torch.cat(true_lngs, 0).numpy()
    true_clusters = torch.cat(true_clusters, 0).numpy()
    pred_clusters = torch.cat(pred_clusters, 0).numpy()
    texts = [item for sublist in texts for item in sublist]
    pred_cluster_ids = [pred_clusters[i].argmax() for i in range(len(pred_clusters))]
    acc = len(true_clusters[true_clusters == pred_cluster_ids]) / len(true_clusters)
    logger.info(
        f'Epoch {epoch} eval loss {te_loss}  accuracy {acc} ' +
        f'true distance avg {true_distance_ls.mean()} true distance median {true_distance_ls.median()}')
    if not no_save:
        model_name = arch if arch is not None else "charcnn"
        if cluster_id is not None:
            filename = f"models/{model_name}-{cluster_id}-{epoch}"
            torch.save(model.fc3, filename)
        else:
            filename = f"models/{model_name}-class-%d" % epoch
            torch.save(model, filename)
        logger.info(f"saved to {filename}")
    return true_lats, true_lngs, true_clusters, pred_clusters, texts


def train_epoch_rel(epoch, model, training_generator, valid_generator, optimizer, args, logger, cluster_id=None,
                    arch=None):
    model.train()
    losses = []
    for iter, batch in tqdm(enumerate(training_generator), total=len(training_generator)):
        model.train()
        feature, label_true, _, lat, lng, langs = batch
        feature = feature.to(device)
        label_true = label_true.to(device)
        langs = langs.to(device)
        optimizer.zero_grad()
        predictions = model(feature, langs)
        loss = custom_loss(predictions, label_true).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if iter % args.log_steps == 0:
            logger.info(f'Epoch {epoch} training loss {sum(losses) / len(losses)}')
            losses = []
        if iter % args.eval_steps == 0 or iter == len(training_generator) - 1:
            validate_rel(epoch=epoch, model=model, valid_generator=valid_generator, logger=logger,
                         cluster_id=cluster_id, arch=arch)

    logger.info(f'Epoch {epoch} training loss {sum(losses) / len(losses)}')


def show_metrics(true_lats, true_lngs, true_clusters, pred_clusters, min_distance, logger, cluster_id=None, all_thresholds=False):
    MAEs = []
    Medians = []
    F1s = []
    percentages = []
    thresholds = [j / 20.0 for j in range(0, 20)] if all_thresholds else [0, 0.75]
    for threshold in thresholds:
        part_true_distance_ls = []
        acc = []
        vals = []
        tp = 0
        fp = 0
        fn = 0
        for i in range(true_clusters.shape[0]):
            pred_cluster_proba = torch.nn.Softmax(dim=0)(torch.tensor(pred_clusters[i])).numpy()
            pred = pred_clusters[i].argmax()
            pred_val = pred_cluster_proba[pred]
            dist = true_distance_from_pred_cluster(pred, true_lats[i], true_lngs[i], cluster_df)
            if dist < min_distance and pred_val >= threshold:
                tp += 1
            elif dist >= min_distance and pred_val >= threshold:
                fp += 1
            elif dist < min_distance and pred_val < threshold:
                fn += 1
            if pred_val < threshold:
                continue
            vals.append(pred_val)
            acc.append(1 if pred == true_clusters[i] else 0)
            part_true_distance_ls.append(dist)
        part_true_distance_ls = pd.Series(part_true_distance_ls)
        MAEs.append(part_true_distance_ls.mean())
        Medians.append(part_true_distance_ls.median())
        percentages.append(len(part_true_distance_ls) / true_clusters.shape[0])
        if tp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = tp / (tp + 0.5 * (fp + fn))
        else:
            precision = 0
            recall = 0
            f1 = 0
        F1s.append(f1)
        logger.info(f'threshold {threshold} MAE {part_true_distance_ls.mean()} ' +
                    f'Median {part_true_distance_ls.median()} ' +
                    f'percentage {len(part_true_distance_ls) / true_clusters.shape[0]} ' +
                    f'acc {pd.Series(acc).mean()} ' +
                    f'precision {precision} recall {recall} ' +
                    f'f1@{min_distance} {f1}')
    if cluster_id is not None:
        with open("clusters.csv", "a") as fout:
            fout.write(
                f'{cluster_id};{MAEs[0]};{MAEs[1]};{Medians[0]};{Medians[1]};{percentages[0]};{percentages[1]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test_input_file', type=str, help='Source csv file')
    parser.add_argument('--train_input_file', type=str, help='Source csv file')
    parser.add_argument('--test_input_file', type=str, help='Source csv file for test')
    parser.add_argument('--max_test', type=int, help='Limit number of testing samples')
    parser.add_argument('--do_train', type=bool)
    parser.add_argument('--do_test', type=bool)
    parser.add_argument('--load_model_dir', type=str, help='Load model from dir and continue training')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=140)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--load_clustering', type=str, help='Load cluster centers, vocab and scaler from directory')
    parser.add_argument('--max_train', type=int, help='Limit number of training samples')
    parser.add_argument('--train_skiprows', type=int, help='Skip first N training samples')
    parser.add_argument('--random_state', type=int, default=300)
    parser.add_argument('--n_conv_filters', type=int, default=256)
    parser.add_argument('--n_fc_neurons', type=int, default=1024)
    parser.add_argument('--eval_batches', type=int, default=32, help='Number of batches for evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--min_distance', type=int, default=500)
    parser.add_argument('--load_first_level_model', type=str,
                        help='Directory of first level model when training second level')
    parser.add_argument('--cluster_id', type=int, help='Train only one second level model for selected cluster')
    parser.add_argument('--arch', type=str, help='Architecture: charcnn or byt5', default='charcnn')
    parser.add_argument('--byt5_model_name', type=str, default='google/byt5-small')
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--use-language', type=str)
    parser.add_argument('--all_thresholds', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=0)
    parser.add_argument('--country_by_coordinates', type=str)
    parser.add_argument('--lang_home_country', type=str)
    parser.add_argument('--external_factor', type=float, default=0.0)
    parser.add_argument('--rare_language_factor', type=float, default=0.0)
    parser.add_argument('--rare_cluster_factor', type=float, default=0.0)
    parser.add_argument('--weight_max', type=float, default=16.0)
    parser.add_argument('--weight_stats', type=str)
    parser.add_argument('--distance_based_smoothing', type=float, default=0.0)
    parser.add_argument('--keep_layer_count', type=int)

    args = parser.parse_args()

    Path('logs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)

    logger = logging.getLogger("")
    logging.basicConfig(level=logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler("logs/debug.log"))

    logger.info("start")

    if args.wandb_project:
        wandb.init(project=args.wandb_project)

    df_train = None

    if args.train_test_input_file is not None:
        df_train, df_test = read_train_test_data(args.train_test_input_file)
        logger.info("finish reading train, test file")

    if args.test_input_file is not None:
        df_test = read_csv_data(args.test_input_file, nrows=args.max_test)
        logger.info("finish reading test file")

    if args.train_input_file is not None:
            df_train = read_csv_data(args.train_input_file, nrows=args.max_train, skiprows=(
            lambda x: x > 0 and x < args.train_skiprows) if args.train_skiprows is not None else None)
            logger.info("finish reading train file")
            print(df_train.columns)

    if args.load_clustering is not None:
        scaler, vocabulary = read_scaler_and_vocab(args.load_clustering)
        with open(args.load_clustering + 'clustering.pkl', 'rb') as fin:
            cluster_df, merges = pickle.load(fin)

    language_df = None
    if args.use_language is not None:
        language_df = pd.read_csv(args.use_language)
        print("language_df", len(language_df))

    tree = BallTree(np.deg2rad(cluster_df[['lat', 'lng']].values), metric='haversine')

    max_length = args.max_length
    device = args.device
    n_clusters_ = len(cluster_df)

    distance_between_clusters = None
    
    sampler = None

    training_params = {"batch_size": args.batch_size,
                        "shuffle": True if sampler is None else False,
                       "num_workers": 0, "sampler": sampler}
    test_params = {"batch_size": args.batch_size,
                   "shuffle": False,
                   "num_workers": 0}
    if args.do_train:
        if args.arch == 'byt5':
            training_set = ByT5ClusteredClassifierDataset(df_train, scaler, args.byt5_model_name, tree, max_length,
                                                          merges=merges)

    if args.arch == 'byt5':
        full_test_set = ByT5ClusteredClassifierDataset(df_test, scaler, args.byt5_model_name, tree, max_length,
                                                       merges=merges)
   
    test_set = torch.utils.data.Subset(full_test_set,
                                       random.choices(range(0, len(df_test)),
                                                      k=args.eval_batches * test_params['batch_size']))
    random.seed(args.random_state)
    valid_set = torch.utils.data.Subset(full_test_set,
                                        random.choices(range(0, len(df_test)), k=test_params['batch_size']))

    
    if args.load_model_dir is not None:
        model = torch.load(args.load_model_dir, map_location=torch.device(device))
        logger.info("model loaded")
    else:  # train from scratch
        if args.arch == 'byt5':
            model = ByT5_classifier(n_clusters=len(cluster_df), model_name=args.byt5_model_name, keep_layer_count=args.keep_layer_count)
            print(model)
        

    if torch.cuda.is_available():
        model.to(device)

    if args.wandb_project:
        wandb.watch(model, log_freq=100)

    test_generator = DataLoader(test_set, **test_params)
    valid_generator = DataLoader(valid_set, **test_params)
    full_test_generator = DataLoader(full_test_set, **test_params)

    custom_loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
   
    if args.arch == 'byt5':
        optimizer = torch.optim.Adam(
            [{'params': model.fc3.parameters(), 'lr': 1e-3}, {'params': model.byt5.parameters(), 'lr': 1e-4}])
  

    if args.do_train:
        training_generator = DataLoader(training_set, **training_params)
        model.train()

        lr_epoch = [1e-3, 1e-4, 1e-5]
        if args.load_first_level_model is not None:
            lr_epoch = [1e-4, 1e-5, 1e-6]
        num_epochs = args.num_epochs
        for epoch in range(args.start_epoch, num_epochs):
            if epoch < len(lr_epoch):
                optimizer.param_groups[0]['lr'] = lr_epoch[epoch]
                for g in optimizer.param_groups:
                    g['lr'] = min(g['lr'], lr_epoch[epoch])
            train_epoch_rel(epoch=epoch, model=model, training_generator=training_generator,
                            valid_generator=test_generator, optimizer=optimizer, args=args, logger=logger,
                            cluster_id=args.cluster_id, arch=args.arch)

    if args.do_test:
        true_lats, true_lngs, true_clusters, pred_clusters, texts = validate_rel(epoch=args.num_epochs, model=model,
                                                                                 valid_generator=full_test_generator,
                                                                                 logger=logger, show_progress=True,
                                                                                 no_save=True,
                                                                                 cluster_id=args.cluster_id,
                                                                                 arch=args.arch)
        test_results_filename = 'test_results.pkl' if args.cluster_id is None else 'test_results-' + str(
            args.cluster_id) + '.pkl'
        with open(test_results_filename, 'wb') as fout:
            pickle.dump((true_lats, true_lngs, true_clusters, pred_clusters, texts), fout)
        show_metrics(true_lats, true_lngs, true_clusters, pred_clusters, args.min_distance, logger,
                     cluster_id=args.cluster_id, all_thresholds=args.all_thresholds)
