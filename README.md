# MSc Thesis by Rok Šikonja

## Install
    
    python -m pip install --upgrade pip
    
    pip install numpy numba matplotlib pandas seaborn PyPDF2 # Basics     
    pip install grid2op

    pip install pyomo tensorflow sklearn

    pip install glpk mosek # Solvers
    pip jupyter

## Tests
    
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
