import json
import os
import time

import torch
from loguru import logger
from timm.utils import AverageMeter

from _data import build_loader, get_topk, get_class_num
from _utils import (
    build_optimizer,
    calc_learnable_params,
    EarlyStopping,
    init,
    mean_average_precision,
    print_in_md,
    save_checkpoint,
    seed_everything,
    validate_smart,
    rename_output,
)
from config import get_config
from loss import COXILoss
from network import build_model


def train_epoch(args, dataloader, net, criterion, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["p_loss", "c_loss", "loss", "mAP"]:
        stat_meters[x] = AverageMeter()

    net.train()
    for images, labels, _ in dataloader:
        images, labels = images.to(args.device), labels.to(args.device)
        embeddings, logits = net(images)

        loss = criterion(embeddings, logits, labels)

        stat_meters["loss"].update(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        q_cnt = labels.shape[0] // 10
        map_v = mean_average_precision(
            embeddings[:q_cnt].sign(), embeddings[q_cnt:].sign(), labels[:q_cnt], labels[q_cnt:]
        )
        stat_meters["mAP"].update(map_v)

        torch.cuda.empty_cache()

    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.1f}]" if "n_" in x else f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )


def train_init(args):
    # setup net
    net, out_idx = build_model(args, True)

    # setup criterion
    criterion = Loss(args).to(args.device)

    logger.info(f"number of learnable params: {calc_learnable_params(net, criterion)}")

    # setup optimizer
    to_optim = [
        {"params": net.parameters(), "lr": args.lr, "weight_decay": args.wd},
        {"params": criterion.parameters(), "lr": args.lr_p},
    ]
    optimizer = build_optimizer(args.optimizer, to_optim)

    return net, out_idx, criterion, optimizer


def train(args, train_loader, query_loader, dbase_loader):
    net, out_idx, criterion, optimizer = train_init(args)

    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, epoch)

        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                out_idx=out_idx,
                parallel_val=args.parallel_val,
            )
            if early_stop:
                break

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"without improvement, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"reach epoch limit, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    return early_stopping.best_epoch, early_stopping.best_map


def prepare_loaders(args, bl_fnc):
    train_loader, query_loader, dbase_loader = (
        bl_fnc(
            args.data_dir,
            args.dataset,
            "train",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
            drop_last=True,
        ),
        bl_fnc(
            args.data_dir,
            args.dataset,
            "query",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        ),
        bl_fnc(
            args.data_dir,
            args.dataset,
            "dbase",
            None,
            batch_size=args.batch_size,
            num_workers=args.n_workers,
        ),
    )
    return train_loader, query_loader, dbase_loader


def main():
    init()
    args = get_config()

    # rename_output(args)

    dummy_logger_id = None
    rst = []
    for dataset in ["cifar", "nuswide", "flickr", "coco"]:
        # for dataset in ["coco"]:
        print(f"processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        train_loader, query_loader, dbase_loader = prepare_loaders(args, build_loader)

        for hash_bit in [16, 32, 64, 128]:
            # for hash_bit in [32]:
            print(f"processing hash-bit: {hash_bit}")
            seed_everything()
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pkl") for x in os.listdir(args.save_dir)):
                print(f"*.pkl exists in {args.save_dir}, will pass")
                continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(
                    vars(args),
                    f,
                    indent=4,
                    sort_keys=True,
                    default=lambda o: o if type(o) in [bool, int, float, str] else str(type(o)),
                )

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})

    print_in_md(rst)


if __name__ == "__main__":
    main()
