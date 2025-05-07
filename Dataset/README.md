---
license: cc-by-sa-4.0
dataset_info:
  features:
  - name: label
    dtype: int64
  - name: content
    dtype: string
  - name: dog_whistle
    dtype: string
  - name: ingroup
    dtype: string
  - name: type
    dtype: string
  splits:
  - name: test
    num_bytes: 204207254
    num_examples: 705815
  - name: train
    num_bytes: 1842627028
    num_examples: 6352333
  - name: train_b
    num_bytes: 8067676
    num_examples: 28900
  - name: test_b
    num_bytes: 915609
    num_examples: 3240
  - name: train_bh
    num_bytes: 7721584
    num_examples: 27640
  - name: test_bh
    num_bytes: 882021
    num_examples: 3111
  - name: train_bhr
    num_bytes: 9433898
    num_examples: 33472
  - name: test_bhr
    num_bytes: 1036523
    num_examples: 3720
  - name: test_hidden
    num_bytes: 227183
    num_examples: 889
  download_size: 1228036796
  dataset_size: 2075118776
---
This dataset merges the datasets from SALT-NLP about dogwhistles:
* [SALT-NLP/silent_signals](https://huggingface.co/datasets/SALT-NLP/silent_signals): Identified dogwhistles
* [SALT-NLP/informal_potential_dogwhistles](https://huggingface.co/datasets/SALT-NLP/informal_potential_dogwhistles): Potential dogwhistles on Reddit
* [SALT-NLP/formal_potential_dogwhistles](https://huggingface.co/datasets/SALT-NLP/formal_potential_dogwhistles): Potential dogwhistles in the US Congress

It adds a ``label`` feature to distinguish between dogwhistles identified by the SALT-NLP team, and not-dogwhistles:
* 0: Not a dogwhistle (absent from SALT-NLP/silent_signals)
* 1: Identified dogwhistle (present in SALT-NLP/silent_signals)

The dataset is organized in four batches. Each batch is splitted in train (90%) and test (10%) splits.
* ``train`` , ``test``: Raw fusion of the SALT-NLP datasets
* ``train_b`` , ``test_b``: Under-sampled from ``train`` , ``test`` to balance ``label``
* ``train_bh`` , ``test_bh``: Hidden dogwhistles (``dog_whistle``) from ``train_b`` , ``test_b`` to test the detection of emergence of dogwhistles. The hidden ``dog_whistle`` represent 10% of the dogwhistles available, and are the following: ``['send me','illegal immigrant','Trilateral Commission','bugman','property rights','Judeo-Christian values','gender abolitionist','young buck','cuckservatives','deadbeat','tax cut','disloyalty','international banks','erasure of women','gender abolitionists','superpredators','bix nood','agender','biological realist','hardworking American','oy vey','wombyn','shekel','colorblind','#alllivesmatter','natal men','physical removal','the Fed' "don't see color" 'personal responsibility','string-pullers','biological men','Judeo-Christian','41']``
* ``train_bhr`` , ``test_bhr``: Under/Over-sampled from ``train_bh`` , ``test_bh`` to balance ``ingroup`` around 2,000 items per group, and then to balance ``label``
* ``test_hidden``: Test split with only items containing dogwhistles that are hidden in the previous splits, to test the detection of emergence of dogwhistles.

The following histograms show the distribution of ``label``, ``ingroup`` and ``type`` in the concatenated ``train_bhr`` and ``test_bhr`` batch.

<img src="stats/dataset-bhr_hist_label.png" alt="Histogram of label in train_bhr+test_bhr" width="500"/>
<img src="stats/dataset-bhr_hist_ingroup.png" alt="Histogram of ingroup in train_bhr+test_bhr" width="500"/>
<img src="stats/dataset-bhr_hist_type.png" alt="Histogram of type in train_bhr+test_bhr" width="500"/>

This dataset was created for the EPFL course EE-559 about Deep Learning.