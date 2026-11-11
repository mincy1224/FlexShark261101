# FlexShark 261101

1. 依赖

   ```sh
   sudo apt-get update
   sudo apt-get install -y build-essential cmake libeigen3-dev zlib1g-dev python3-venv
   python -m pip install -r requirements.txt datasets onnx
   ```

2. 编译
   ```sh
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel
   ```

3. 预处理
   
   这是预处理2组 Bert，先看看结果
   ```sh
   python script/bert_pretreat.py --root bert_data --weights /path/to/bert_weights_fixed.npz --count 2
   ```

   完整实验
   ```sh
   python script/bert_pretreat.py --root bert_data --weights /path/to/bert_weights_fixed.npz
   ```

   这是预处理20组 LeNet-5，先看看结果
   ```sh
   python script/lenet5_pretreat.py --root lenet5_data --count 20
   ```

   完整实验
   ```sh
   python script/lenet5_pretreat.py --root lenet5_data --count
   ```

4. 实验
   Bert
   ```sh
   python script/bert_test.py --root bert_data
   ```
   LeNet-5
   ```sh
   python script/lenet5_test.py --root lenet5_data
   ```
