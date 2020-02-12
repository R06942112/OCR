#!/usr/env/bin python3
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display_step',type=int,default=100,help='display step')
    parser.add_argument('--save_step',type=int,default=1000,help='save step')
    parser.add_argument('--iteration',type=int,default=300000,help='iteration')
    parser.add_argument('--max_gradient_norm',type=float,default=5.0,help='max gradient norm')
    parser.add_argument('--batch_size',type=int,default=128,help='batch size')
    parser.add_argument('--height',type=int,default=100,help='image height')
    parser.add_argument('--width',type=int,default=100,help='image width')
    parser.add_argument('--channels',type=int,default=1,help='image channel')
    parser.add_argument('--encoder_length',type=int,default=23,help='encoder length')
    parser.add_argument('--decoder_length',type=int,default=30,help='decoder length')
    parser.add_argument('--tgt_vocab_size',type=int,default=38,help='target vocabulary')
    parser.add_argument('--embedding_size',type=int,default=256,help='embedding size')
    parser.add_argument('--num_units',type=int,default=256,help='rnn unit size')
    parser.add_argument('--beam_width',type=int,default=3,help='beam width')
    parser.add_argument('--test_txt',type=str,default='../dataset/iiit5k.txt',help='txt file of testing images')
    parser.add_argument('--lex_txt',type=str,default='../dataset/iiit5k_lex.txt',help='txt file of testing lexicon')
    parser.add_argument('--save_dir',type=str,default='./save_dir/model.ckpt',help='model saved path')
    parser.add_argument('--load_dir',type=str,default='./model/',help='pertrained weight path')
    parser.add_argument('--train_txt',type=str,default='../dataset/synth90k.txt',help='txt file of training images')
    flags, _ = parser.parse_known_args()

    return flags

    if not os.path.exists(flags.save_dir.rsplit('/',1)[0]):
        os.makedirs(flags.save_dir.rsplit('/',1)[0])


if __name__ == '__main__':
    args = parse_args()

