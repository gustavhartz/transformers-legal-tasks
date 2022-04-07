# this scripts downloads multiple gigabytes of CUAD relared data from the CUAD website and related links
# Prompt user to accept download of data
#
# Usage:
#   ./getData.sh
#

# Ask user to accept download
echo "This script will download data from the CUAD website.  This data is used to create the CUAD database."
echo "1"
# Ask user to input yes to continue
echo "Do you wish to continue? (y/n) "
read REPLY

# Check cwd is this directory
if [[ $PWD != *"data" ]]; then
    echo "Please run this script from the data directory"
    exit 1
fi


if [[ $REPLY =~ ^[Yy]$ ]]
then
    wget https://raw.githubusercontent.com/TheAtticusProject/cuad/main/category_descriptions.csv
    wget https://github.com/TheAtticusProject/cuad/blob/main/data.zip?raw=true -O data.zip
    unzip data.zip
    rm data.zip
    mv data/* .
    rm -r data

    # Get the roberta chekpoint from atticus
    # if roberta-base folder exists, skip
    if [ ! -d "roberta-base" ]; then
        wget https://zenodo.org/record/4599830/files/roberta-base.zip?download=1 -O roberta-base.zip
        unzip roberta-base.zip
        rm roberta-base.zip
    fi


    # Get remaining files from atticus
    # if CUAD_v1 folder exists, skip
    if [ ! -d "CUAD_v1" ]; then
        wget https://zenodo.org/record/4599830/files/CUAD_v1.zip?download=1 -O CUAD_v1.zip
        unzip CUAD_v1.zip
        rm CUAD_v1.zip
    fi

else
    echo "Download aborted."
fi
