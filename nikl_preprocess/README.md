# Preparation for Korean speech

## Corpus
https://github.com/homink/speech.ko

## Command

### Multi-speaker
```
cd nikl_preprocess
python prepare_metadata.py --corpus ${corpus location} --trans_file ${corpus location}/trans.txt --spk_id ${corpus location}/speaker.mid
```
### Single-speaker
Default single speaker id is fv01 written in ${corpus location}/speaker.sid. You can edit speaker.sid.
