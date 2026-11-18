import os
files = os.listdir('/workspace/dota_ori/DOTA_raw/val/labelTxt/')
with open("val.txt", "w") as f:
    for p in files:
        f.write(p.strip()[:-4]+"\n")