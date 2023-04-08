#!/bin/bash 
parser_repo="https://github.com/Milzi/arguEParser"
model_repo="https://github.com/negedng/argument_BERT"
student_corpora="https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2422/ArgumentAnnotatedEssays-2.0.zip?sequence=1&isAllowed=y"


wget -O student_corpora.zip $student_corpora
unzip student_corpora.zip -d student_corpora
unzip student_corpora/ArgumentAnnotatedEssays-2.0/brat-project-final.zip -d essays
rm student_corpora.zip

git clone $parser_repo
mkdir arguEParser/inputCorpora/essays
mv essays/brat-project-final/* essays/
rmdir essays/brat-project-final
mv essays arguEParser/inputCorpora/
cd arguEParser
python CorpusParserStandalone.py