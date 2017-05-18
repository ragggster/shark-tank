f_in = open("season4ordered.csv",'r')
prevep = 0
numpitch = 0
for line in f_in.readlines():
    split = line.split(",")
    season = split[0] 
    ep = int(split[1])
    amt = split[3]
    valuation = split[4] 
    if (split[2] == 'No'): 
    	write = "0 0 0"
    else: 
    	write = "1 " + str(amt) + " " + str(valuation) 
    if (ep == prevep): 
    	numpitch+= 1
    else: 
    	numpitch = 1
    	prevep += 1
    filepath = "s" + str(season) + "-e" + str(ep) + "-p" + str(numpitch) + ".txt"
    # print filepath 
    # print write 
    # print "\n"
    text_file = open(filepath, "w")
    text_file.write(write)
    text_file.close()

f_in.close()
