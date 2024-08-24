dataset=$1
device=$2

[ -z "${dataset}" ] && dataset="synthetic"
[ -z "${device}" ] && device=-1


python -m pdb run_transformer.py \
	--device $device \
	--dataset $dataset 
