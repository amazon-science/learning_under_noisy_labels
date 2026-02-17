# When Annotators Disagree: A Principled Approach to Learning with Noisy Labels

The code is written in Python 3.

### Dependencies

Install the Python3 dependecies by executing the following command:
```
pip3 install -r requirements.txt
```

### Tests
In the root folder you can run some sanity check tests, by executing the following command:
```
bash run_tests.sh
```
### To run Text Classification experiments:
```
cd examples/text_experiments 
```

```
bash run_text_exp.sh
```

### To run TrashNet experiments:
```
cd data && git clone https://github.com/garythung/trashnet.git
```

```
mv trashnet/data/dataset-resized.zip . && rm -rf trashnet && unzip dataset-resized.zip
```

```
cd ../examples/trashnet_experiments && python3 generate_synthetic_annotations.py
```

```
bash run_trashnet_exp.sh
```

### To run experiments on CIFAR-10N:
```
cd examples/cifar10n_experiments 
```

```
bash run_cifar_exp.sh
```

### To run synthetic experiments:
```
cd examples/syntethic_experiments 
```

```
bash run_exp.sh
```

For doubts or errors feel free to ping purificato@diag.uniroma1.it!

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the Apache 2.0 License.

## Acknowledgments

The implementation of Dawid-Skene and Iterative-Weighted Majority Voting draws from thethe paper [A Lightweight, Effective, and Efficient Model for LabelAggregation in Crowdsourcing](https://github.com/yyang318/LA_onepass). We gratefully acknowledge the authors for making their code available.
