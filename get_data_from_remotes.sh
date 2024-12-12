#!/bin/bash
# to
# 0.5
#scp -P 50434 -r ./*.{py,txt,sh} ./hparams ./algos/*.py root@178.74.56.47:.
# 2.0
#scp -P 14478 -r ./*.{py,txt,sh} ./hparams ./algos/*.py root@81.166.173.12:.
# 3.0
#scp -P 39583 -r ./*.{py,txt,sh} ./hparams ./algos/*.py root@193.143.121.66:.
# from
#
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
    log="$(date +"%Y-%m-%d %H:%M:%S") $(echo "saving ${values[2]} runs from port: ${values[0]} ip: ${values[1]}")"
    echo $log
    echo $log >> scp_to_vastai.log
    scp -P ${values[0]} -r ${values[1]}:runs ${values[2]}_runs/
done < "$input_file"

#scp -P 50434 -r root@178.74.56.47:runs vast_humanoid_0.5_runs/
#scp -P 14478 -r root@81.166.173.12:runs vast_humanoid_2.0_runs/
#scp -P 39583 -r root@193.143.121.66:runs vast_humanoid_3.0_runs/

#
# mkdir algos
# mv Shaped* algos/
# mv Models.py algos/
# mv BaseAgent.py algos/
# sudo chmod +x ./start_vastai_runs.sh
# ./start_vastai_runs.sh "Humanoid-v4" "td3" 2.0