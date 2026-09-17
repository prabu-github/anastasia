# anastasia
Code for anastasia

---

# Data setup: Data download and make compatible

- `cd projworks`
- `python anastasia/datasetup.py --extension csv`
- `python anastasia/datasetup.py --extension parquet`
- CSV is for humans, parquet is for speed! 


---

# Test `project.py`
- Data setup assumed done.
- `cd projworks`
- `python anastasia/test_project.py --get_xtransforms`
- `python anastasia/test_project.py --get_ytransforms`
- `python anastasia/test_project.py --get_modelnames --pattern '*'`
- `python anastasia/test_project.py --get_data --modelname dplsr__anastasia-mc_nitrogen__full-uv__id`
- `python anastasia/test_project.py --train --modelname dplsr__anastasia-mc_nitrogen__full-uv__id`
- `python anastasia/test_project.py --train --modelname univar__anastasia-mc_nitrogen__full-uv__id`

---

# CHTC setup: Create submit files

- `cd projworks`
- `python anastasia/chtcsetup.py --models_per_submit 1`

---

# Note for `run.py`
- `cd projworks`
- Two ways to specify `--train_model`:
   - `python anastasia/run.py --train_model MODELNAME`
   - `python anastasia/run.py --train_model MODELNAME1:MODELNAME2:MODELNAME3:...`
    
---

# CHTC

### local
- `cd projworks`
- `tar --exclude hytraits/test -czf  forchtc/hytraits.tar.gz hytraits/`
- `tar --exclude anastasia/assets --exclude anastasia/origdata --exclude anastasia/gendata -czf forchtc/anastasia.tar.gz anastasia`

### staging
- `hytraits-cpu.sif` must be in `anastasia/toremote`
- Clean `anastasia/fromremote` as needed!

### submit
- Clean `/home/pravindran/anastasia`

### local to staging
- `cd projworks`
- `scp forchtc/hytraits.tar.gz forchtc/anastasia.tar.gz pravindran@transfer.chtc.wisc.edu:/staging/p/pravindran/anastasia/toremote`

### local to submit
- `cd projworks`
- `scp anastasia/gendata/forchtc/* pravindran@townsend-ap.chtc.wisc.edu:/home/pravindran/anastasia`

--- 

# After training

### staging to local
- `cd projworks`
- `scp pravindran@transfer.chtc.wisc.edu:/staging/p/pravindran/anastasia/fromremote/<PATTERN>.tar.gz .`

### Extract tar.gzs
- `for file in *.tar.gz; do tar -xzf "$file"; done`
---






