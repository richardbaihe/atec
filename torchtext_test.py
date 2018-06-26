from torchtext import data, datasets
import os

DATA = os.path.abspath('./data')
# TRANS = datasets.snli.ShiftReduceField()
TRANS = datasets.snli.ParsedTextField()

train = data.TabularDataset(path=os.path.join(DATA,'trans.txt'),
                    format='tsv',
                    fields=[
                        ('trans', TRANS)
                    ])

TRANS.build_vocab(train)

train_iter = data.Iterator(train, batch_size=1,
                           sort=False, repeat=False)

for sample in train_iter:
    print(sample.trans)
