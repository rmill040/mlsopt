#!/usr/bin/env bash

##################################################################
# Boostrap script to install software on AWS and run experiments #
##################################################################

echo "updating settings"
sudo yum update -y

echo "installing anaconda"
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
sh *.sh -b

echo "installing git"
sudo yum install git -y

echo "cloning mlsopt repository and installing package"
git clone https://github.com/rmill040/mlsopt.git
cd mlsopt
~/anaconda3/bin/pip install -e .

echo "upgrade core packages"
~/anaconda3/bin/pip install scikit-learn numpy pandas hyperopt --upgrade

echo "running experiments"
~/anaconda3/bin/python paper/src/experiment_no_tpe.py

echo "stoping this EC2 instance"
shutdown -h now

echo "all done!"