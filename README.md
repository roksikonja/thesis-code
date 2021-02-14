# MSc Thesis by Rok Šikonja

    git clone https://github.com/roksikonja/thesis-code.git

## Install
    
    python -m pip install --upgrade pip
    
    pip install numpy numba matplotlib pandas seaborn PyPDF2 # Basics     
    pip install grid2op

    pip install pyomo
    pip install tensorflow sklearn

    pip install glpk mosek gurobi # Solvers
    
    pip jupyter pygame
    pip install tensorflow-addons
    pip install graph_nets "tensorflow>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability --use-feature=2020-resolver
    
    pip install l2rpn-baselines
    pip install imageio
        
    pip freeze > requirements-win-10.txt
    pip freeze > requirements-ubuntu-20-04-LTS.txt

    
## Units

[Units and conversion](https://pandapower.readthedocs.io/en/v2.2.2/elements/line.html)


## Tests
    black . --exclude="./venv"
    
    python -m unittest discover <directory>
    python -m unittest discover tests
    
## GLPK

### Windows
    
    # Install Microsoft Visual Studio C++ 14.0
    # http://www.osemosys.org/uploads/1/8/5/0/18504136/glpk_installation_guide_for_windows10_-_201702.pdf
    # http://winglpk.sourceforge.net/
    # Download latest 
    # https://sourceforge.net/projects/winglpk/files/latest/download
    # https://sourceforge.net/projects/winglpk/
    # Unzip and copy to C:\glpk-X.Y
    # Add C:\glpk-X.Y\w64 to System PATH
    glpsol --help  # Check
   
    
### Linux WSL
    
    # Fix issue - Build manually
    sudo add-apt-repository ppa:rafaeldtinoco/lp1871129
    sudo apt update
    sudo apt install libc6=2.31-0ubuntu8+lp1871129~1 libc6-dev=2.31-0ubuntu8+lp1871129~1 libc-dev-bin=2.31-0ubuntu8+lp1871129~1 -y --allow-downgrades
    sudo apt-get install liblzma-dev
    sudo apt-mark hold libc6
    
    # Ubuntu
    sudo apt-get install glpk-utils
    glpsol --version  # Check
   
    cd /mnt/c/Users/chrosik/data_grid2op
    cp -R rte_case5_example ~/data_grid2op/rte_case5_example
    cp -R l2rpn_2019 ~/data_grid2op/l2rpn_2019
   
    cd /mnt/c/Users/chrosik
    cp -R -v data_grid2op ~/

    
# Deep Mind Graph Networks

    pip install grid2op matplotlib numba
    pip install jupyter
    pip install graph_nets "tensorflow>=2.1.0-rc1" "dm-sonnet>=2.0.0b0" tensorflow_probability --use-feature=2020-resolver


# Microsoft GNNs

    https://github.com/microsoft/tf2-gnn

    pip install tf2_gnn
    
    python -m pip install --upgrade pip
    pip check  # Check dependecy conflicts
    
    pip install <package> --use-feature=2020-resolver
    
# TensorBoard

    tensorboard --logdir=/full_path_to_your_logs
    
# GPU
    High scale NNs
    Look ahead multiple steps ahead
    
    15 minutes-generator schedules
    
    # S
    
# Colab

    https://github.com/roksikonja/thesis-code
