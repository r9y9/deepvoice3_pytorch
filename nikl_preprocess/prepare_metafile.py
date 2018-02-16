from __future__ import print_function
import subprocess,os,re

def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def pe(cmd, shell=False):
    """
    Print and execute command on system
    """
    ret = []
    for line in execute(cmd, shell=shell):
        ret.append(line)
        print(line, end="")
    return ret


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Produce metafile where wav file path and its transcription are aligned",
                                   epilog="Example usage: python preprea_metadata $HOME/copora/NIKL")
  parser.add_argument("--corpus_dir", "-c",
                      help="filepath for the root directory of corpus",
                      required=True)

  parser.add_argument("--trans_file", "-t",
                      help="Extracted transcription file obatained from extract_trans.py",
                      required=True)

  parser.add_argument("--spk_id", "-sid",
                      help="Speaker ID for single speaker such as fv01",
                      required=False)
  args = parser.parse_args()

  print("Prepare metadata file for all speakers")
  pe("find %s -name %s | grep -v 'Bad\|Non\|Invalid' > %s/wav.lst" % (args.corpus_dir,"*.wav",args.corpus_dir),shell=True)

  trans={}
  with open(args.trans_file,"r") as f:
    for line in f:
      line = line.rstrip()
      line_split = line.split(" ")
      trans[line_split[0]] = " ".join(line_split[1:])

  with open(args.corpus_dir+"/wav.lst", "r") as f:
    wavfiles = f.readlines()

  pe("rm -f %s/metadata.txt" % (args.corpus_dir),shell=True)
  for w in wavfiles:
    w = w.rstrip()
    tid = re.search(r'(t[0-9][0-9]_s[0-9][0-9])',w)
    if tid:
      tid_found = tid.group(1)
      pe('echo %s"|"%s >> %s/metadata.txt' % (w,trans.get(tid_found),args.corpus_dir),shell=True)

  print("Metadata files is created in %s/metadata.txt" % (args.corpus_dir))
  pe("ls -d -- %s/*/ | grep -v 'Bad\|Non\|Invalid' | rev | cut -d'/' -f2 | rev > %s/speaker.mid" % (args.corpus_dir,args.corpus_dir),shell=True)
  pe("head -n 1 %s/speaker.mid > %s/speaker.sid" % (args.corpus_dir,args.corpus_dir),shell=True)
