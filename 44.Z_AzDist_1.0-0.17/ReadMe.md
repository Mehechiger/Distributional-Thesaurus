**What is AzDist?**
AzDist is a distributionnal thesaurus realized as part of the L3LI projet in Université Paris-Diderot. It reads CoNLL format corpora and genrates automatically a thesaurus according to the distribution of words in various contexts.

**Prerequisites**
Hardware: AzDist is designed to be functionnal under any modern computer with a minimum of live memories of 4GB (for CoNLL format corpora of around 5GB). Note that for certain features such as 3 and above grams mode, more live memories are required.
Software dependencies: AzDist requires a certain number of python libraries installed, in particular: tqdm, SciPy, NumPy, NumExpr, Pandas.

**Installation**
Just decompress the archive in a folder.

**Quick start**
Go into the program folder under the command line tool, and simply run it with:
`python3 AzDist.py --d *DIRECTORY_OF_CONLL_CORPORA*`
Detailed progress is given on screen. After a few minutes (depending on your computer's hardwares), the program will notify you of it's completion. Some csv files such as "AzDist_A.csv" should appear in the program folder, where "A" indicates the "adjective" part of speech, which would be your fresh baked distributionnal thesaurus (by part of speech).

**Advanced parameters**
Advanced parameters could be passed to AzDist under command line while launching:

| PARAMETERS AND ACCEPTABLE VALUES                             | COMMENTS                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `-h` <br />`--help`                                          | show help message and exit                                   |
| `-d CORPSDIR`<br />`--corpsdir CORPSDIR`                     | corpora directory                                            |
| `-ps {{'V'},{'N'},{'ADV'},{'A'},{'ADV', 'N'},{'N', 'V'},{'ADV', 'A'},{'N', 'A'},{'V', 'A'},{'ADV', 'V'},{'ADV', 'N', 'A'},{'ADV', 'N', 'V'},{'N', 'V', 'A'},{'ADV', 'V', 'A'},{'ADV', 'N', 'V', 'A'}}`<br />`--poss {{'V'},{'N'},{'ADV'},{'A'},{'ADV', 'N'},{'N', 'V'},{'ADV', 'A'},{'N', 'A'},{'V', 'A'},{'ADV', 'V'},{'ADV', 'N', 'A'},{'ADV', 'N', 'V'},{'N', 'V', 'A'},{'ADV', 'V', 'A'},{'ADV', 'N', 'V', 'A'}}` | PoSs to be analized (use comma, to seperate PoSs and do not insert space between them, eg. python3 AzDist.py - c A, V); default is A,ADV,N,V |
| `-m {['2', 'gram'],['3', 'gram'],['4', 'gram'],['5', 'gram'],['6', 'gram'],['7', 'gram'],['8', 'gram'],['9', 'gram'],dep}`<br />`--mode {['2', 'gram'],['3', 'gram'],['4', 'gram'],['5', 'gram'],['6', 'gram'],['7', 'gram'],['8', 'gram'],['9', 'gram'],dep}` | contexts extract mode(specify the n in case of ngram, use dash - to seperate n and gram and do not insert space between them, eg. "python3 AzDist -m 5-gram"); default is 2-gram |
| `-wf {RelFreq,TTest,PMI}`<br />`--wfunc {RelFreq,TTest,PMI}` | weight function; default is TTest                            |
| `-sf {Cosine,Jaccard,Lin}`<br />`--sfunc {Cosine,Jaccard,Lin}` | similariy function; default is Cosine                        |
| `-s SIZE`<br />`--size SIZE`                                 | number of similar words under one item; default is 10        |
| `-p PATH`<br />`--path PATH`                                 | set output (abs) path; default is path of AzDist.py          |

**Authors**
A.V.T.A.
Z.F.

**Licence**
MIT
