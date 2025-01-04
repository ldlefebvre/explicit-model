To run: python3 train_model.py
To clear caches: sudo sync && sudo sysctl -w vm.drop_caches=3
To see gpu usage: watch -n 1 nvidia-smi
To see memory usage: watch -n 1 free -h
htop
cat output2.log | tee temp.txt

Other useful commands:
mkdir -p dataset2/{train,test}/{nsfw,safe} && cp -r dataset/train/{hentai,porn} dataset2/train/nsfw && cp -r dataset/train/{neutral,drawings,sexy} dataset2/train/safe && cp -r dataset/test/{hentai,porn} dataset2/test/nsfw && cp -r dataset/test/{neutral,drawings,sexy} dataset2/test/safe

mkdir -p dataset2/{train,test}/{nsfw,safe} && \
rsync -a --ignore-errors dataset/train/hentai/ dataset2/train/nsfw/ | pv -lep -s $(find dataset/train/hentai/ -type f | wc -l) > /dev/null 2>&1 & \
rsync -a --ignore-errors dataset/train/porn/ dataset2/train/nsfw/ | pv -lep -s $(find dataset/train/porn/ -type f | wc -l) > /dev/null 2>&1 & \
rsync -a --ignore-errors dataset/train/neutral/ dataset2/train/safe/ | pv -lep -s $(find dataset/train/neutral/ -type f | wc -l) > /dev/null 2>&1 & \
rsync -a --ignore-errors dataset/train/drawings/ dataset2/train/safe/ | pv -lep -s $(find dataset/train/drawings/ -type f | wc -l) > /dev/null 2>&1 & \
rsync -a --ignore-errors dataset/train/sexy/ dataset2/train/safe/ | pv -lep -s $(find dataset/train/sexy/ -type f | wc -l) > /dev/null 2>&1 & \
rsync -a --ignore-errors dataset/test/hentai/ dataset2/test/nsfw/ | pv -lep -s $(find dataset/test/hentai/ -type f | wc -l) > /dev/null 2>&1 & \
rsync -a --ignore-errors dataset/test/porn/ dataset2/test/nsfw/ | pv -lep -s $(find dataset/test/porn/ -type f | wc -l) > /dev/null 2>&1 & \
rsync -a --ignore-errors dataset/test/neutral/ dataset2/test/safe/ | pv -lep -s $(find dataset/test/neutral/ -type f | wc -l) > /dev/null 2>&1 & \
rsync -a --ignore-errors dataset/test/drawings/ dataset2/test/safe/ | pv -lep -s $(find dataset/test/drawings/ -type f | wc -l) > /dev/null 2>&1 & \
rsync -a --ignore-errors dataset/test/sexy/ dataset2/test/safe/ | pv -lep -s $(find dataset/test/sexy/ -type f | wc -l) > /dev/null 2>&1 & \
wait

nohup python3 train_model.py > output.log 2>&1 &
ps aux | grep train_model.py
tail -f output.log
reptyr 13067
kill 13067

tmux new -s train_model
python3 train_model.py 2>&1 | tee output.log
Detach: Press Ctrl+b followed by d.
Reattach: tmux attach -t train_model
tail -f output.log
tmux ls
tmux kill-session -t train_model

scp wsl:/home/lagoupo/code/ldlefebvre/explicit-model/dataset/test/drawings/004f0b56b85378df6a31813cb08b676e812b6e4c7a424b422936bebb341405d2.jpg .
open 004f0b56b85378df6a31813cb08b676e812b6e4c7a424b422936bebb341405d2.jpg




Copy all images from remote desktop to mac in the misclassified_image.csv with:
ssh -M -S ~/.ssh/ssh_mux_wsl -fN wsl
-M: Enables master mode for connection sharing.
-S ~/.ssh/ssh_mux_wsl: Specifies the control socket path.
-f: Requests SSH to go to the background after authentication.
-N: Indicates that no command will be executed on the remote system.

ssh -S ~/.ssh/ssh_mux_wsl wsl "tail -n +2 /home/lagoupo/code/ldlefebvre/explicit-model/misclassified_images.csv" \
| while IFS=, read -r file_name pred_label true_label; do
    # Create the target directory locally if it doesn't exist
    mkdir -p "${true_label}"

    # Define the full remote path to the image
    remote_path="/home/lagoupo/code/ldlefebvre/explicit-model/dataset/test/${file_name}"

    # Use rsync with the control socket to copy the image
    rsync -e "ssh -S ~/.ssh/ssh_mux_wsl" -av "wsl:${remote_path}" "${true_label}/"
done

ssh -S ~/.ssh/ssh_mux_wsl -O exit wsl


OR


Step 1: Edit Your SSH Config File
Add the following to your ~/.ssh/config file:

Host wsl
    ControlMaster auto
    ControlPath ~/.ssh/ssh_mux_%h_%p_%r
    ControlPersist yes
ControlMaster auto: Enables automatic connection sharing.
ControlPath: Specifies the path for the control socket.
ControlPersist yes: Keeps the master connection open in the background after the initial session is closed.

ssh -M -S ~/.ssh/ssh_mux_wsl -fN wsl

ssh wsl "tail -n +2 /home/lagoupo/code/ldlefebvre/explicit-model/misclassified_images.csv" \
| while IFS=, read -r file_name pred_label true_label; do
    mkdir -p "${true_label}"
    remote_path="/home/lagoupo/code/ldlefebvre/explicit-model/dataset/test/${file_name}"
    rsync -av "wsl:${remote_path}" "${true_label}/"
done

ssh wsl "exit"
ssh -S ~/.ssh/ssh_mux_wsl -O exit wsl
