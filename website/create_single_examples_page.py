#!/usr/bin/env python

import glob
import sys
import os

def main():
    fn = sys.argv[1]
    f = open(fn, "r")
    ln = f.readlines()
    f.close()

    out = ""
    for l in ln:
        if l.startswith("INSERT "):
            ff = l.replace("INSERT ", "").strip()
            raw_rst = "/".join(fn.split("/")[:-1]).strip() + "/all_examples/" + ff.replace(".py", ".rst")
            if os.path.exists(raw_rst) == False:
                out += "\n\n"
            else:
                out += "\n\n\n" + simplify_one_rst(raw_rst) + "\n\n\n"
        else:
            out += l

    print(out)
    
def simplify_one_rst(fn):
    f = open(fn, "r")
    ln = f.readlines()
    f.close()

    start = None
    for i, l in enumerate(ln):
        if list(set(l.strip())) == ["="]:
            ln[i] = ln[i].replace("=", "-")
            start = i - 1 - 3
    if start is None:
        print("Problem")
        exit()

    ln = ln[start:]

    ln_new = []
    i = 0
    while i < len(ln):
        l = ln[i]
        if l.strip().startswith(".. container:: sphx-glr-download sphx-glr-download-jupyter"):
            i = i + 4
            continue
        ln_new.append(l)
        i = i + 1
    ln = ln_new
    
    out = ""
    use = True
    for l in ln:
        if use == False and not l == "\n" and not l == "|\n" and not l.startswith(" "):
            use = True
        if l.startswith(".. GENERATED FROM PYTHON SOURCE") or \
           l.startswith(".. rst-class:: sphx-glr-timing") or \
           l.startswith("   **Total running time of the script:**") or \
           l.startswith(".. rst-class:: sphx-glr-script-out"):
            use = False
        if use == True:
            if ":class: sphx-glr-multi-img" in l:
                l = l.replace(":class: sphx-glr-multi-img", ":class: sphx-glr-single-img")
            if l.strip().startswith(":download:"):
                l = l.replace(" <example_", " <all_examples/example_")
            out += l.rstrip() + "\n"
            
    while "\n\n\n" in out:
        out = out.replace("\n\n\n", "\n\n")

    return out

main()
