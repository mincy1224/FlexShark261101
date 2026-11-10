# FlexShark 261101


## LeNet-5
Download the weights & dataset.

```shell
python3 -m pip install onnx

chmod +x ./script/lenet5_data_ready.sh
./script/lenet5_data_ready.sh
```

Test The Application.

```shell
python3 ./script/lenet5_test.py
```

```shell
OMP_NUM_THREADS=8 python3 ./script/lenet5_test.py --binary build-rel/run_lenet5
```

The run reports two metrics per sample set:

- `acc` — the secret-shared prediction against the MNIST ground-truth label;
- `agree` — the secret-shared prediction against the plaintext prediction of
  the original model (`lenet5.onnx`), which `lenet5_data_ready.sh` computes up
  front with `script/lenet5_plaintext.py` and stores as `plaintext.csv`.