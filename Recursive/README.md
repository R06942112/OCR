Recursive Recurrent Nets with Attention Modeling for OCR in the Wild
======================================

This software implements the [Recursive Recurrent Nets with Attention Modeling for OCR in the Wild](https://arxiv.org/abs/1603.03101).
The code is modified from [da03](https://github.com/da03/Attention-OCR)

Pretrained model can be found in [link](https://drive.google.com/open?id=1kKMGcxeEEJiZ6F79l1qbO3o7apErvOkQ)

Test
--------
    python test.py [test_txt] [lex_txt]
	
[test_txt] is the path to the annotation (eg., './svt.txt')

[lex_txt] is the path to the lexicon (eg., './svt_lex.txt')
	
Train
--------
    python test.py [train_txt] 
	
[train_txt] is the path to the annotation (eg., './synth90k.txt')




