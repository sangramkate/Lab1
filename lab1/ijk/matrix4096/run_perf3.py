import os
import subprocess

import csv

from math import ceil

#Add prefertch stats",

stat_list_mini = ["cycles"]

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
  "fp_arith_inst_retired.scalar_single",
  "cycle_activity.stalls_l1d_miss",
  "cycle_activity.stalls_l2_miss",
  "cycle_activity.stalls_mem_any",
  "cycle_activity.stalls_total",
]

"""
Memory_BW:
  MLP
	[Memory-Level-Parallelism (average number of L1 miss demand load when there is at least 1 such miss)]
Memory_Bound:
  Load_Miss_Real_Latency
	[Actual Average Latency for L1 data-cache miss demand loads]
  MLP
	[Memory-Level-Parallelism (average number of L1 miss demand load when there is at least 1 such miss)]
Memory_Lat:
  Load_Miss_Real_Latency
	[Actual Average Latency for L1 data-cache miss demand loads]

Summary:
  GFLOPs
	[Giga Floating Point Operations Per Second]
"""

#TODO: How do you use these summary stats?",
for k in range(1,8):
    for i in range(int(ceil(len(stat_list)/5))):
        outputfile = "matrix4096_small/list3_" + str(i)
        event_str = stat_list[5*i] + ":u," + stat_list[5*i+1] + ":u," + stat_list[5*i+2] + ":u," + stat_list[5*i+3] + ":u," + stat_list[5*i+4] + ":u"
        check_err = subprocess.Popen(['perf','stat','-e',event_str,'-x',',','-o',outputfile,'--append','./mm4096'], stdout=subprocess.PIPE)
        output = check_err.communicate()[0]
        output = (output.decode('utf8').strip())	


#for i in range(len(stat_list)):
#    print(stat_list[i])
#    event_str = stat_list[i] + ":u"

