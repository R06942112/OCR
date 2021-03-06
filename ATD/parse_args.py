#!/usr/env/bin python3
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--display_step',type=int,default=100,help='display step')
    parser.add_argument('--save_step',type=int,default=1000,help='save step')
    parser.add_argument('--iteration',type=int,default=300000,help='iteration')
    parser.add_argument('--learning_rate',type=float,default=0.01,help='learning rate')
    parser.add_argument('--max_gradient_norm',type=float,default=5.0,help='max gradient norm')
    parser.add_argument('--batch_size',type=int,default=256,help='batch size')
    parser.add_argument('--height',type=int,default=32,help='image height')
    parser.add_argument('--width',type=int,default=100,help='image width')
    parser.add_argument('--channels',type=int,default=1,help='image channel')
    parser.add_argument('--encoder_length',type=int,default=25,help='encoder length')
    parser.add_argument('--decoder_length',type=int,default=30,help='decoder length')
    parser.add_argument('--tgt_vocab_size',type=int,default=39,help='target vocabulary')
    parser.add_argument('--embedding_size',type=int,default=256,help='embedding size')
    parser.add_argument('--num_units',type=int,default=256,help='rnn unit size')
    parser.add_argument('--beam_width',type=int,default=5,help='beam width')
    parser.add_argument('--r_path',type=str,default='./model/recognizer/',help='recognition model path')
    parser.add_argument('--c_path',type=str,default='./model/classifier/',help='typeset classifier model path')
    parser.add_argument('--test_txt',type=str,default='../dataset/iiit5k.txt',help='txt file of testing images')
    parser.add_argument('--lex_txt',type=str,default='../dataset/iiit5k_lex.txt',help='txt file of testing lexicon')
    parser.add_argument('--save_dir',type=str,default='./save_dir/model.ckpt',help='model saved path')
    parser.add_argument('--train_s_txt',type=str,default='../dataset/synth90k.txt',help='txt file of testing images')
    parser.add_argument('--train_u_txt',type=str,default='../dataset/synth_ENGV.txt',help='txt file of testing images')
    flags, _ = parser.parse_known_args()
    
    if not os.path.exists(flags.save_dir.rsplit('/',1)[0]):
        os.makedirs(flags.save_dir.rsplit('/',1)[0])


    return flags


if __name__ == '__main__':
    args = parse_args()

