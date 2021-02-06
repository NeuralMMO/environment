echo "Neural MMO setup; assumes Anaconda Python 3.7 and gcc"

if [[ $1 == "--SERVER_ONLY" ]]; then 
   echo "You have chosen not to install the graphical rendering client"
elif [[ $1 == "" ]]; then
   echo "Installing Neural MMO Client (Unity3D)..."
   git clone --depth=1 https://github.com/jsuarez5341/neural-mmo-client
   mv neural-mmo-client forge/embyr
else
   echo "Specify either --SERVER_ONLY or no argument"
   exit 1
fi

#Install python packages
pip install -r scripts/requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
ray install-nightly
pip install ray[rllib]
echo "Errors upon pip install ray[rllib] are normal. If the environment runs, setup is correct"
