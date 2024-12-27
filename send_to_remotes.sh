#!/bin/bash
# to
# 0.5
#scp -P 50434 -r ./*.{py,txt,sh} ./hparams ./algos/*.py root@178.74.56.47:.
# 2.0
#scp -P 14478 -r ./*.{py,txt,sh} ./hparams ./algos/*.py root@81.166.173.12:.
# 3.0
#
# from
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

# Input file
input_file="$1"

# Iterate through each line of the file
while IFS= read -r line; do
    # Split the line into positional values using space as a delimiter
    IFS=',' read -r -a values <<< "$line"
    # Print the values as a comma-separated string
    log="$(date +"%Y-%m-%d %H:%M:%S") $(echo "sending code to port: ${values[0]} ip: ${values[1]}")"
    echo $log
    echo $log >> scp_to_remote.log
    scp -P ${values[0]} -r ./*.{py,txt,sh} ./hparams ./algos/*.py ${values[1]}:.
done < "$input_file"
