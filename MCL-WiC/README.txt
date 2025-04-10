=================================================================================
  SemEval-2021 Task 2: Multilingual and Cross-lingual Word-in-Context (MCL-WiC)
            
        Federico Martelli, Najla Kalach, Gabriele Tola and Roberto Navigli
                Sapienza NLP Group, Sapienza University of Rome

                  Github: https://github.com/SapienzaNLP/mcl-wic
           Codalab: https://competitions.codalab.org/competitions/27054                                                                                                                                                                     
=================================================================================

Multilingual and Cross-lingual Word-in-Context Disambiguation (MCL-WiC) is the
first SemEval task for Word-in-Context disambiguation which tackles the
challenge of capturing the polysemous nature of words without relying on a fixed
sense inventory in a multilingual and cross-lingual setting. MCL-WiC provides a 
single high-quality framework for the performance evaluation of a wide range of 
approaches aimed at evaluating the capability of a system to deeply understand 
word meaning.

=================================================================================
FORMAT
=================================================================================

The data is available in JSON format. Two types of file are available: the .data 
and the .gold file. The former contains id, lemma, POS, sentence1, sentence2, 
and offsets, whereas the latter contains id and answer. The answer is either 
T (True) or F (False) depending on whether the target word occurring in two 
sentences has the same meaning or not. 

=================================================================================
CONTENT
=================================================================================

The training folder (training) contains the following files:
multilingual/
    training.en-en.data
    training.en-en.gold

The development folder (dev) contains the following files:
multilingual/
    dev.ar-ar.data
    dev.ar-ar.gold
    dev.en-en.data
    dev.en-en.gold
    dev.fr-fr.data
    dev.fr-fr.gold
    dev.ru-ru.data
    dev.ru-ru.gold
    dev.zh-zh.data
    dev.zh-zh.gold

The test folder (test) contains the following files:    
multilingual/
    test.ar-ar.data
    test.en-en.data
    test.fr-fr.data
    test.ru-ru.data
    test.zh-zh.data
crosslingual/
    test.en-ar.data
    test.en-fr.data
    test.en-ru.data
    test.en-zh.data

=================================================================================
LICENSE
=================================================================================

The data is released under the CC-BY-NC 4.0 license (see LICENSE.txt). If you use
the data, please put a link to this repo and cite the paper below.

=================================================================================
REFERENCE
=================================================================================

F. Martelli, N. Kalach, G. Tola, R. Navigli. SemEval-2021 Task 2: Multilingual 
and Cross-lingual Word-in-Context Disambiguation (MCL-WiC). Proc. of 15th 
Workshop on Semantic Evaluation.

=================================================================================
ACKNOWLEDGMENTS
=================================================================================

The authors gratefully acknowledge the support of the ELEXIS EU Project No. 731015 
and the ERC Consolidator Grant MOUSSE No. 726487 under the European Union's 
Horizon 2020 research and innovation programme.

=================================================================================
CONTACTS
=================================================================================

If you have any enquiries, please contact Federico Martelli and Roberto Navigli
at <surname>@di.uniroma1.it.

