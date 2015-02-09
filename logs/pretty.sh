#/bin/bash

FN=`find -type f -printf '%T+ %p\n' | sort -r | head -n 1 | grep -oP " (.)+log"`
FN=${FN:3}

paste <(cat "$FN" | grep 'Epochs seen' | grep -oP "\d\d:\d\d:\d\d") <(cat "$FN" | grep 'Epochs seen' | grep -oP "\d+$") <(cat "$FN" | grep train_y_nll | grep -oP '\d.\d+$' ) <(cat "$FN" | grep valid_y_nll | grep -oP '\d.\d+$')
