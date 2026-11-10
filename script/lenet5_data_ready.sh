DATA_DIR=./lenet5_data
COUNT="${1:-999}"
mkdir -p "$DATA_DIR"
curl -fL https://sbondaryev.dev/models/lenet5.onnx -o "$DATA_DIR/lenet5.onnx"
python3 script/lenet5_weight_format.py --root "$DATA_DIR"
python3 script/lenet5_mnist_format.py --root "$DATA_DIR" --download --count "$COUNT"
python3 script/lenet5_plaintext.py --root "$DATA_DIR" --limit "$COUNT"
