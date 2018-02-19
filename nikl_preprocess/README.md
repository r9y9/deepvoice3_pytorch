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
```
cd nikl_preprocess
python prepare_metadata.py --corpus ${corpus location} --trans_file ${corpus location}/trans.txt --spk_id ${corpus location}/speaker.sid
```
Default single speaker id is fv01. You can edit it by speaker id in [here](https://github.com/homink/speech.ko).
