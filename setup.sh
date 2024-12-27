apt-get install libglfw3-dev libgles2-mesa-dev
pip install setuptools==65.5.0
pip install swig
pip install -r requirements.txt

# move the scp-ed files to the right place
 mkdir algos
 mv Shaped* algos/
 mv Models.py algos/
 mv BaseAgent.py algos/
 sudo chmod +x ./start_runs.sh