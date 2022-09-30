import configargparse


def get_args_parser():
    parser = configargparse.ArgumentParser(add_help=False)
    # config
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--name', type=str)

    # data
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--data_type', type=str)
    parser.add_argument('--num_classes', type=int)

    # vis
    parser.add_argument('--visdom_port', type=int, default=8097)
    parser.add_argument('--vis_step', type=int, default=100)

    # model
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--img_size', type=int)
    parser.add_argument('--patch_size', type=int)
    parser.add_argument('--in_chans', type=int)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--mlp_ratio', type=float)

    # # data augmentation
    # # * Mixup params
    # parser.add_argument('--mixup_beta', type=float, default=0.8, help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    # parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    # parser.add_argument('--mix_prob', default=0.5, type=float, help='mixup probability')
    # parser.add_argument('--switching_prob', type=float, default=0.5,
    #                     help='Probability of switching to cutmix when both mixup and cutmix enabled')
    # parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # train
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--save_step', type=int, default=40, help='if save_step < epoch, then save')
    parser.add_argument('--num_workers', type=int, default=0)

    # FIXME
    parser.set_defaults(distributed=False)
    parser.add_argument('--gpu_ids', nargs="+")
    parser.add_argument('--rank', type=int)
    parser.add_argument('--world_size', type=int)

    # FIXME 10 to 1000
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=5e-5)  # 5e-2 # 1e-2 # 5e-3 # 1e-3 # 5e-4 # 1e-4
    parser.add_argument('--log_dir', type=str, default='./logs')

    parser.add_argument('--test_epoch', type=str, default='best')
    return parser

