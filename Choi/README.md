Simultaneous Recognition of Horizontal and Vertical Text in Natural Images
======================================

This software implements the [Simultaneous Recognition of Horizontal and Vertical Text in Natural Images](https://arxiv.org/abs/1812.07059)

Pretrained model can be found in [link](https://drive.google.com/open?id=1vs1_ggZru8u2cSvqKlMsYOlsAEjF2RRH)

Test
--------
    python test.py [test_txt] [lex_txt]
	
[test_txt] is the path to the annotation (eg., './svt.txt')

[lex_txt] is the path to the lexicon (eg., './svt_lex.txt')
	
Train
--------
    python test.py [train_txt] 
	
[train_txt] is the path to the annotation (eg., './synth90k.txt')




