import yaml
import argparse
import os
from easydict import EasyDict
from interfaces.super_resolution import TextSR
from setup import Logger


def main(config, args):
    Mission = TextSR(config, args)
    if args.test:
        if args.image_path != None:
            Mission.inference(args.image_path)
        else:
            Mission.test()
    else:
        Mission.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='super_resolution.yaml')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--STN', action='store_true', default=True, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--mask', action='store_true', default=True, help='')
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--test_model', type=str, default='CRNN', choices=['ASTER', "CRNN", "MORAN",'ABINet','MATRN','PARSeq'])
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--lr_position', type=float, default=1e-4, help='fine tune for position aware module')
    parser.add_argument('--image_path', type=str)
    args = parser.parse_args()
    config_path = os.path.join('config', args.config)
    config = yaml.load(open(config_path, 'rb'), Loader=yaml.Loader)
    config = EasyDict(config)
    config.TRAIN.lr = args.learning_rate
    parser_TPG = argparse.ArgumentParser()
    Logger.init('logs', 'LEMMA', 'train')
    Logger.enable_file()
    main(config, args)
