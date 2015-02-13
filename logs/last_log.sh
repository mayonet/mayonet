#bin/bash

echo `find -type f -printf '%T+ %p\n' | sort -r | head -n 1 | grep -oP "\./(.)+\.log"`
