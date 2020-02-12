Learning to Read Irregular Text with Attention Mechanisms
======================================

This software implements the [Learning to Read Irregular Text with Attention Mechanisms](http://www.cse.psu.edu/~duk17/papers/IJCAI17-IrregularText.pdf).
The code is modified from [da03](https://github.com/da03/Attention-OCR)

Pretrained model can be found in [link](https://drive.google.com/open?id=1YHg9cfUVQHXncW6z7NxhqYndV6aHoS1U)

Test
--------
    python test.py [test_txt] [lex_txt]
	
[test_txt] is the path to the annotation (eg., './svt.txt')

[lex_txt] is the path to the lexicon (eg., './svt_lex.txt')
	
Train
--------
    python test.py [train_txt] 
	
[train_txt] is the path to the annotation (eg., './synth90k.txt')



