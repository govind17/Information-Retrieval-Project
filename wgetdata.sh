#!/bin/bash


python trans_data.py train.txt mq2008.train mq2008.train.group

python trans_data.py test.txt mq2008.test mq2008.test.group

python trans_data.py vali.txt mq2008.vali mq2008.vali.group