
import utils
import utils2

path = '/data/jstrassb/SuperParsing/SiftFlow/Experiments'
eaw_path = '/data/jstrassb/SuperParsing/SiftFlow/eaw'
fl_comb = [[0,1,2,3],[0,1,2],[0,1,3],[0,2,3],[1,2,3],[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]
bias = [[1,0,-1,-2],[1,0,-1],[1,0,-1],[1,0,-1],[1,0,-1],[1,0],[1,0],[1,0],[1,0],[1,0],[1,0]]
f = 1
f2 = [1,2,3,4]
f1 = [8,4,2,1]


results_sem1 = []
results_geo1 = []

results_sem2 = []
results_geo2 = []

results_sem3 = []
results_geo3 = []

for i in range(len(fl_comb)):
	fl = fl_comb[i]
	ip = utils.eaw_val1(path, eaw_path, fl)
#	ip2 = utils2.eaw_val1(path, eaw_path, fl)
	
	f2 = [2**j for j in fl]
	f1 = [8.0/(2**j) for j in fl]

	results_sem1 += [utils.eaw_val2(ip,0,f,bias[i])]
	results_sem2 += [utils.eaw_val2(ip,1,f2,bias[i])]
	results_sem3 += [utils.eaw_val2(ip,0,f1,bias[i])]
	
#	results_geo1 += [utils2.eaw_val2(ip2,0,f,bias[i])]
#	results_geo2 += [utils2.eaw_val2(ip2,1,f2,bias[i])]
#	results_geo3 += [utils2.eaw_val2(ip2,0,f1,bias[i])]

print 'sem 0:'
print results_sem1
print 'geo 0:'
print results_geo1

print 'sem 1:'
print results_sem2
print 'geo 1:'
print results_geo2
	
print 'sem 2:'
print results_sem3
print 'geo 2:'
print results_geo3
