input_files = ["matrix32/cycles.txt","matrix512/cycles.txt","matrix4096/cycles.txt","matrix8192/cycles.txt"]
outputfile = "cycles.txt"


out_file = open(outputfile,"w")
for inputfilename in input_files:
    new_file = open(inputfilename,"r")
    for f in new_file:
        out_file.write(f)
out_file.close()
