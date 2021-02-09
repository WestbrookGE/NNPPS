****In case that you have trouble running this code, we have provided the expected output in the 'result.png' file.****

This is the code for the algorithm proposed in our paper "To Be or not to Be, Tail Labels in Extreme Multi-label Learning". Thanks for paying attention to our work.

About N^2P^2S
=============
N^2P^2S is proposed to measure whether a tail label should be removed in XML scenarios. There are two hyper-parameters: tail bound(t) and cut rate(\delta). Labels with label frequency(positive instances appeared in training data) less than tail bound and N^2P^2S less than cut rate will be removed in our method. We suggest setting t = 10 and \delta = 0.1 by default.

Usage
=====
The code is written in Python3. Please make sure you have set up these packages in the newest version:
numpy(pip install numpy)
scipy(pip install scipy)
tqdm(pip install tqdm)
sklearn(pip install sklearn)
xclib(
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python3 setup.py install --user
)
The content inside () is the command for setting up these packages.



Sample Command
--------
python3 NNPP_cut.py $train_X_path $train_lbl_path $test_lbl_path $save_trn_lbl $save_tst_lbl $save_cut_idx $tail_bound $cut_rate



Toy Example
===========
We provide a demo for our method. Please execute "bash demo.sh" (Linux) or "demo" (Windows) in the topmost folder. 


Due to the file size limit, you have to download dataset from XML repository by yourself and split it to feature file and label file. Run the following command to convert file form:

perl convert_format.pl [repository data file] [output feature file name] [output label file name]

Thanks again for paying attention to our work!