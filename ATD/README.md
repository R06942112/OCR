A Scene Text Recognition System of Both Sideways and Upright Text in Arbitrary Orientation
======================================

This software implements the [A Scene Text Recognition System of Both Sideways and Upright Text in Arbitrary Orientation](https://drive.google.com/open?id=1jJks6Yot7e1P1XisPjol721vsg4pd__4)

Pretrained model can be found in [link](https://drive.google.com/open?id=1JCYqx2IaQGjYQz4eeEGuwLnaXtyDCM-E)

Test
--------
    python test.py [test_txt] [lex_txt]
	
[test_txt] is the path to the annotation (eg., './svt.txt')

[lex_txt] is the path to the lexicon (eg., './svt_lex.txt')
	
Train
--------
    python test.py [train_s_txt] [train_u_txt] 
	
[train_s_txt] is the path to the annotation of a sideways dataset (eg., './synth90k.txt')

[train_u_txt] is the path to the annotation of an upright dataset(eg., './synth_ENGV.txt')


