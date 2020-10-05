#! /bin/bash

envname="beat_allies"

echo "Updating conda if necessary"
conda update -n base conda
conda config --set show_channel_urls True

echo "This will create a python env named 'beat_allies' and install the necessary tools for LLMT"
read -n 1 -p "Proceed ([y]/n)? " choice
case "$choice" in 
  y|Y ) printf "\nLet's go!";;
  n|N ) printf "\nQuitting..."; exit 1;;
  * ) printf "\ninvalid input"; exit 1;;
esac


#sed -i 's/ENVNAME/$envname' environment.yml
conda env create -f environment.yml
#conda init bash
source activate $envname

echo "Installing third party tools: nmtpytorch"
read -n 1 -p "Proceed ([y]/n)? " choice
case "$choice" in 
  y|Y ) printf "\nLet's go!";;
  n|N ) printf "\nQuitting..."; exit 1;;
  * ) printf "\ninvalid input"; exit 1;;
esac

cd nmtpytorch
python setup.py develop
cd ..

echo "Installing OpenKiwi"
cd openkiwi
python setup.py
cd ..

echo "You can run the baseline system with the following command:"
echo "beat --prefix `pwd`/beat exp run loicbarrault/loicbarrault/translation_ll_dev/1/translation_ll_dev"
echo "(be sure to export CUDA_VISIBLE_DEVICES=X where X is the GPU you want to use)"

