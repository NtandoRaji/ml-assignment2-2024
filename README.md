# ml-assignment2-2024

## Install Apptainer
I do not recommend using Windows with apptainer. You should use Ubuntu. If you are using Windows use a virtual machine (e.g. VirtualBox) to use Ubuntu. 

N.B. If you are using an MSL machine skip this step.

#### Run the following commands and follow supplied instructions:
- sudo apt update
- sudo apt install -y software-properties-common
- sudo add-apt-repository -y ppa:apptainer/ppa
- sudo apt update
- sudo apt install -y apptainer

## Install Singularity Environment
Download 'rail.sif' from https://drive.google.com/file/d/1i6l2MT5N1-4kOIqf2qVX1kUQUOmNTqL_/view?usp=sharing

## Run Singularity Environment
- Run the following in the same folder as rail.sif "apptainer shell rail.sif"
- In this shell you will now see 'Apptainer>'.
- N.B. While in this shell you must train and save your models.

Evaluate Model
- While NOT IN the singularity environment, navigate to the evaluation folder in your shell.
    - run the command 'apptainer exec rail.sif python3 evaluate.py'
