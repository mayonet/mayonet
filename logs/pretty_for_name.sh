#/bin/bash

FN=$1

tail -fn 100000 $FN | grep "validation_result\|Epochs seen\|train_y_nll\|valid_y_nll"
