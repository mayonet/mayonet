#/bin/bash

FN=$1

paste <(cat "$FN" | grep 'Epochs seen' | grep -oP "\d\d:\d\d:\d\d") <(cat "$FN" | grep 'Epochs seen' | grep -oP "\d+$") <(cat "$FN" | grep train_y_nll | grep -oP '\d.\d+$' ) <(cat "$FN" | grep valid_y_nll | grep -oP '\d.\d+$')
