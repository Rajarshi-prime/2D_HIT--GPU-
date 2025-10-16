#!/bin/bash

for i in 0.70 0.72 0.75 0.77 0.80 0.85 0.90 1.00; do
    echo "alpha = $i"
    ffmpeg -framerate 10 -start_number 100 -i prtcls_time_%d_alph_$i.png \
        -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2:(ow-iw)/2:(oh-ih)/2" \
        -c:v libx265 -crf 20 -pix_fmt yuv420p -movflags +faststart \
        prtcls_$i.mp4 -y
done
