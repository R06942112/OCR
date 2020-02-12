AON
======================================

This software implements the [AON: Towards Arbitrarily-Oriented Text Recognition](https://arxiv.org/abs/1711.04226), modified from [HuiZhang](https://github.com/huizhang0110/AON)

Pretrained model can be found in [link](https://drive.google.com/open?id=12o1H5hmHABRtWolnv0yxFIKlQx4BrK8B)

Test
--------
    python test.py [test_txt] [lex_txt]
	
[test_txt] is the path to the annotation (eg., './svt.txt')

[lex_txt] is the path to the lexicon (eg., './svt_lex.txt')
	
Train
--------
    python test.py [train_txt] 
	
[train_txt] is the path to the annotation (eg., './synth90k.txt')



