import copy
import warnings

warnings.filterwarnings("ignore")

import os
import time

import torch
import torch.distributed as dist
from torch.nn.utils import remove_spectral_norm

import utils


def inner_optimization(G_full_clone, D_full_clone, inner_optimizer_G, inner_optimizer_D, meta_batch_size, data, use_gan_training=False, ngpus=1):
    # forward + backward + optimize
    support_losses, support_generated = G_full_clone(data, stage="Full")
    support_loss = sum([val.mean() for val in support_losses.values()])

    G_full_clone.zero_grad()
    support_loss.backward()
    for param in G_full_clone.parameters():
        if param.grad is not None:
            torch.nan_to_num(param.grad, nan=0, posinf=1e3, neginf=-1e3, out=param.grad)
    torch.nn.utils.clip_grad_norm_(G_full_clone.parameters(), 0.3)
    inner_optimizer_G.step()

    losses_G, loss_G = utils.reduce_loss_dict(support_losses, ngpus), utils.reduce_loss(support_loss, ngpus)
    losses_D, loss_D = {}, 0

    if use_gan_training:
        inner_optimizer_D.zero_grad()
        losses_D = D_full_clone(data, support_generated)
        loss_D = sum([val.mean() for val in losses_D.values()])
        loss_D.backward()
        for param in D_full_clone.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        torch.nn.utils.clip_grad_norm_(D_full_clone.parameters(), 0.3)
        inner_optimizer_D.step()
        losses_D, loss_D = utils.reduce_loss_dict(losses_D, ngpus), utils.reduce_loss(loss_D, ngpus)
    return losses_G, loss_G, losses_D, loss_D, support_generated


def train_ddp(args, conf, models, datasets):
    G = models["generator"]
    D = models["discriminator"]
    if args["local_rank"] == 0:
        print(G)
        print(D)

    dataset_train = datasets["dataset_train"]
    optim_G, optim_D, scheduler_G, scheduler_D = utils.build_outer_optimizer_and_scheduler(conf, G, D)

    if conf["model"].get("warp_ckpt", None):
        utils.load_ckpt(
            conf["model"]["warp_ckpt"],
            {"generator": G},
            device=torch.device("cpu"),
            strict=True,
            warp_ckpt=True,
        )
    if args["ckpt"] is not None:
        start_epoch = utils.load_ckpt(
            args["ckpt"],
            models,
            device=torch.device("cuda", args["device"]),
            strict=False,
        )
        if args["remove_sn"]:
            for _, m in G.named_modules():
                if hasattr(m, 'weight_u') and hasattr(m, 'weight_v'):
                    remove_spectral_norm(m)
    
    G_full, D_full = utils.build_full_model(conf, args, G, D)
    G_full_clone, D_full_clone = utils.clone_model(conf, args, copy.deepcopy(G), copy.deepcopy(D))
    
    start_epoch = 0
    meta_batch_size = conf['dataset'].get('meta_batch_size', 0)
    num_support_samples = conf['dataset'].get('num_support_samples', 1)
    use_gan_training = conf['train']['loss_weights']['generator_gan'] != 0

    if args["fp16"]:
        scaler = torch.cuda.amp.GradScaler()

    if conf["train"]["tensorboard"] and args["local_rank"] == 0:
        event_path = os.path.join(
            conf["train"]["event_save_path"], conf["general"]["exp_name"]
        )
        writer = utils.Visualizer(event_path)

    dtime = 0
    total_iters = 0
    if args["local_rank"] == 0:
        print("start to train...")
    
    for epoch in range(start_epoch + 1, conf["train"]["epochs"] + 1):
        stime = stime_iter = time.time()
        for i, data in enumerate(dataset_train):
            total_iters += 1
            dtime += time.time() - stime
            step = (epoch - 1) * len(dataset_train) + i
            loss_G_inner_init, loss_D_inner_init = 0, 0
            loss_G_inner_last, loss_D_inner_last = 0, 0
            losses_D, loss_D = {}, 0

            optim_G.zero_grad()
            if args["stage"] != "Warp":
                optim_D.zero_grad()
            
            if args["task"] == "Pretrain":
                with torch.cuda.amp.autocast():
                    losses_G, generated = G_full(data, stage=args["stage"])
                    loss_G = sum([val.mean() for val in losses_G.values()])
                scaler.scale(loss_G).backward()
                scaler.unscale_(optim_G)
                for param in G_full.module.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                torch.nn.utils.clip_grad_norm_(G_full.parameters(), 0.3)
                scaler.step(optim_G)
                scaler.update()

                if args["stage"] != "Warp":
                    optim_D.zero_grad()
                    with torch.cuda.amp.autocast():
                        losses_D = D_full(data, generated)
                        loss_D = sum([val.mean() for val in losses_D.values()])
                    scaler.scale(loss_D).backward()
                    scaler.unscale_(optim_D)
                    for param in D_full.module.parameters():
                        if param.grad is not None:
                            torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                    torch.nn.utils.clip_grad_norm_(D_full.parameters(), 0.3)
                    scaler.step(optim_D)
                    scaler.update()
            
            elif args["task"] == "Meta":               
                # Reptile implementation
                support_set, _ = data[0], data[1]   # support_set samples of shape (num_samples_per_cls, num_cls, *)
                data = support_set
                orig_state_dict_G = G_full.module.generator.state_dict()
                orig_state_dict_D = D_full.module.discriminator.state_dict()
                grad_list_G = [torch.zeros(p.size()).to(args["device"]) for p in G_full.module.generator.parameters()]
                grad_list_D = [torch.zeros(p.size()).to(args["device"]) for p in D_full.module.discriminator.parameters()]
                
                for cls_idx in range(conf['dataset'].get('num_classes_per_set', 1)):
                    G_full_clone.generator.load_state_dict(orig_state_dict_G)
                    if use_gan_training:
                        D_full_clone.discriminator.load_state_dict(orig_state_dict_D)

                    support_set_cls = utils.return_cls_data(support_set, cls_idx) # Return data of one cls of shape (num_samples_per_cls, *)
                    inner_optimizer_G, inner_optimizer_D = utils.build_inner_optimizer(conf, G_full_clone, D_full_clone)
                    for inner_step in range(conf['dataset'].get('inner_update_steps', 2)):
                        # Return batch data of shape (meta_bs, *)
                        batch_data = utils.convert_data_to_cuda(utils.return_batch_data(
                                                                    support_set_cls, 
                                                                    (inner_step * meta_batch_size) % num_support_samples, 
                                                                    ((inner_step + 1) * meta_batch_size) % num_support_samples if ((inner_step + 1) * meta_batch_size) % num_support_samples != 0 else num_support_samples
                                                                ))
                        losses_G, loss_G, losses_D, loss_D, support_generated = inner_optimization(G_full_clone, D_full_clone, inner_optimizer_G, inner_optimizer_D, meta_batch_size, batch_data, use_gan_training, args["ngpus"])
                        if inner_step == 0:
                            if cls_idx == 0:
                                generated = support_generated
                            loss_G_inner_init += loss_G / conf['dataset'].get('num_classes_per_set', 1)
                            loss_D_inner_init += loss_D / conf['dataset'].get('num_classes_per_set', 1)
                    
                    loss_G_inner_last += loss_G / conf['dataset'].get('num_classes_per_set', 1)
                    loss_D_inner_last += loss_D / conf['dataset'].get('num_classes_per_set', 1)
                    
                    # Update meta grad
                    for idx, (orig_param, updated_param) in enumerate(zip(G_full.module.generator.parameters(), G_full_clone.generator.parameters())):
                        grad_list_G[idx] += (orig_param.data - updated_param.data) / conf['dataset'].get('num_classes_per_set', 1)

                    if use_gan_training:
                        for idx, (orig_param, updated_param) in enumerate(zip(D_full.module.discriminator.parameters(), D_full_clone.discriminator.parameters())):
                            grad_list_D[idx] += (orig_param.data - updated_param.data) / conf['dataset'].get('num_classes_per_set', 1)
                
                # Sync grad
                for idx in range(len(grad_list_G)):
                    dist.all_reduce(grad_list_G[idx])
                
                if use_gan_training:
                    for idx in range(len(grad_list_D)):
                        dist.all_reduce(grad_list_D[idx])
                
                # Computer grad for outer optimization
                for idx, p in enumerate(G_full.module.generator.parameters()):
                    if p.grad is None:
                        p.grad = torch.zeros(p.size()).to(args["device"])
                    p.grad.data.add_(grad_list_G[idx] / args["ngpus"])

                for param in G_full.module.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=1e3, neginf=-1e3, out=param.grad)
                torch.nn.utils.clip_grad_norm_(G_full.parameters(), 0.3)
                optim_G.step()
                G_full._sync_params_and_buffers()
                
                data = utils.process_vis_data(data)
                
                if use_gan_training:
                    # Computer grad for outer optimization
                    for idx, p in enumerate(D_full.module.discriminator.parameters()):
                        if p.grad is None:
                            p.grad = torch.zeros(p.size()).to(args["device"])
                        p.grad.data.add_(grad_list_D[idx] / args["ngpus"])

                    for param in D_full.module.parameters():
                        if param.grad is not None:
                            torch.nan_to_num(param.grad, nan=0, posinf=1e3, neginf=-1e3, out=param.grad)
                    torch.nn.utils.clip_grad_norm_(D_full.parameters(), 0.3)
                    optim_D.step()
                    D_full._sync_params_and_buffers()
            else:
                raise ValueError("{} task is not defined.".format(args["task"]))

            if args["task"] != "Eval" and args["local_rank"] == 0 and i % conf["train"]["print_freq"] == 0:
                string = "Epoch {} Iter {} D/Time : {:.3f}/{} ".format(epoch, i, time.time() - stime_iter, time.strftime("%Hh%Mm%Ss", time.gmtime(time.time() - stime)))
                for loss_name in losses_G.keys():
                    if losses_G[loss_name].mean() != 0:
                        string += loss_name + " : {:.2f};".format(losses_G[loss_name].mean())
                if args["stage"] != "Warp":
                    for loss_name in losses_D.keys():
                        if losses_D[loss_name].mean() != 0:
                            string += loss_name + " : {:.2f};".format(losses_D[loss_name].mean())
                string += "loss_G_init : {:.2f};loss_D_init : {:.2f}".format(loss_G_inner_init, loss_D_inner_init)
                string += "loss_G_last : {:.2f};loss_D_last : {:.2f}".format(loss_G_inner_last, loss_D_inner_last)
                print(string)
                stime_iter = time.time()

            # Save tensorboard event
            if args["task"] != "Eval" and args["local_rank"] == 0 and step % conf["train"]["event_save_freq"] == 0:
                if conf["train"]["tensorboard"]:
                    print("Epoch {} Iter {} Step {} event save".format(epoch, i, step))
                    utils.save_events(writer, losses_G, loss_G_inner_last, step, losses_D, loss_D_inner_last, generated, loss_G_inner_init, loss_D_inner_init)
                num = min(6, generated["prediction"].shape[0])
                utils.save_training_images(conf, args, writer, generated, data, step, num)

            # Save ckpt
            if (args["task"] != "Eval" and args["local_rank"] == 0 and total_iters % conf["train"]["ckpt_save_iter_freq"] == 0):
                print("save ckpt")
                ckpt_out_path = os.path.join(
                    conf["train"]["ckpt_save_path"], conf["general"]["exp_name"]
                )
                ckpt_models = {
                    "generator": G,
                    "discriminator": D,
                    "optimizer_generator": optim_G,
                    "optimizer_discriminator": optim_D,
                }
                utils.save_ckpt(
                    out_path=ckpt_out_path,
                    epoch=epoch,
                    models=ckpt_models,
                    total_iters=total_iters,
                )

        scheduler_G.step()
        if args["stage"] != "Warp":
            scheduler_D.step()

        # Save checkpoint
        if args["local_rank"] == 0 and epoch % conf["train"]["ckpt_save_freq"] == 0:
            print("save ckpt")
            ckpt_out_path = os.path.join(conf["train"]["ckpt_save_path"], conf["general"]["exp_name"])
            ckpt_models = {
                "generator": G,
                "discriminator": D,
                "optimizer_generator": optim_G,
                "optimizer_discriminator": optim_D,
            }
            utils.save_ckpt(out_path=ckpt_out_path, epoch=epoch, models=ckpt_models)
    return
