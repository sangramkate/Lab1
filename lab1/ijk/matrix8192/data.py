from math import floor
filename1 = "matrix8192_small/list4_0"
filename2 = "matrix8192_small/list4_1"
filename3 = "matrix8192_small/list4_2"
filename4 = "matrix8192_small/list4_3"
filename5 = "matrix8192_small/list4_3"
outputfile = "output.txt"
resultfile = "result.txt"
cyclefile = "cycles.txt"

stat_list = [
  "cpu-cycles",
  "L1-dcache-load-misses",
  "L1-dcache-loads",
  "L1-dcache-stores",
  "L1-icache-load-misses",
  "LLC-load-misses",
  "LLC-loads",
  "LLC-store-misses",
  "LLC-stores",
  "l2_rqsts.all_demand_data_rd",
  "l2_rqsts.all_pf",
  "l2_rqsts.pf_miss",
  "l2_rqsts.all_demand_miss",
  "l2_rqsts.all_demand_references",
  "l2_rqsts.references",
  "fp_arith_inst_retired.scalar_single",
  "fp_arith_inst_retired.scalar_double",
  "fp_arith_inst_retired.128b_packed_double",
  "fp_arith_inst_retired.128b_packed_single",
  "fp_arith_inst_retired.256b_packed_double",
  "fp_arith_inst_retired.256b_packed_single",
  "fp_arith_inst_retired.512b_packed_double",
  "fp_arith_inst_retired.512b_packed_single",
  "fp_arith_inst_retired.scalar_double",
]

data = {}

for event in stat_list:
    data[event] = 0

count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0

file2 = open(cyclefile,"w")
for event in stat_list[0:4]:
    file1 = open(filename1,"r")
    for f in file1:
       if event in f:
           data[event] = data[event] + int(f.split(',')[0])
           if event == "cpu-cycles":
		file2.write(str(float(f.split(',')[0])/(2.1 * 1000000000)) + "\n")
       if "#" in f:
           count0 = count0 + 1 
    data[event] = int(floor(data[event]/count0))
file2.close()

for event in stat_list[5:9]:
    file2 = open(filename2,"r")
    for f in file2:
       if event in f:
           data[event] = data[event] + int(f.split(',')[0])
       if "#" in f:
           count1 = count1 + 1
    data[event] = int(floor(data[event]/count1))

for event in stat_list[10:14]:
    file3 = open(filename3,"r")
    for f in file3:
       if event in f:
           data[event] = data[event] + int(f.split(',')[0])
       if "#" in f:
           count2 = count2 + 1
    data[event] = int(floor(data[event]/count2))

for event in stat_list[15:19]:
    file4 = open(filename4,"r")
    for f in file4:
       if event in f:
           data[event] = data[event] + int(f.split(',')[0])
       if "#" in f:
           count3 = count3 + 1
    data[event] = int(floor(data[event]/count3))
     
for event in stat_list[20:24]:
    file5 = open(filename5,"r")
    for f in file5:
       if event in f:
           data[event] = data[event] + int(f.split(',')[0])
       if "#" in f:
           count4 = count4 + 1
    data[event] = int(floor(data[event]/count4))

new_file = open(outputfile,'w')
for event in stat_list:
    new_line = event + ":" + str(data[event]) + "\n"
    new_file.write(new_line)
new_file.close()
new_file = open(resultfile, 'w')
new_file.write("execution time:")
new_line = str((float(data["cpu-cycles"])/float(2.10 * 1000000000)))
new_file.write(new_line + "\n")
new_file.write("L1 miss rate:")
new_line = str((float(data["L1-dcache-load-misses"])/float((data["L1-dcache-loads"] + data["L1-dcache-stores"]))))
new_file.write(new_line + "\n")
new_file.write("L2 miss rate:")
new_line = str(float(data["l2_rqsts.all_demand_miss"])/float(data["l2_rqsts.all_demand_references"]))
new_file.write(new_line + "\n")
new_file.write("L3 miss rate:")
new_line = str(float((data["LLC-load-misses"]+data["LLC-store-misses"]))/float((data["LLC-loads"]+data["LLC-stores"])))
new_file.write(new_line + "\n")
new_file.close()

