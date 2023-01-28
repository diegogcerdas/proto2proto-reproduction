git clone git@github.com:archmaester/proto2proto.git proto2proto_tmp

cp -r proto2proto_tmp/lib reproduction
rm -rf proto2proto_tmp

mkdir tmp_data/CUB_200_2011
python setup_cub.py
