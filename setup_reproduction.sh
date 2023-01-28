git clone git@github.com:archmaester/proto2proto.git proto2proto_tmp

cp -r proto2proto_tmp/lib reproduction
rm -rf proto2proto_tmp

mkdir -p ./tmp_data/CUB_200_2011
python setup_cub.py

mkdir -p ./datasets/CUB_200_2011
cp -r tmp_data/CUB_200_2011/dataset/train_crop ./datasets/CUB_200_2011
cp -r tmp_data/CUB_200_2011/dataset/test_crop ./datasets/CUB_200_2011
rm -rf ./tmp_data
