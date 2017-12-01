testFile = "bayesbest.py"
trainDir = "movies_reviews/"
testDir = "db_txt_files/"

execfile(testFile)
bc = Bayes_Classifier(trainDir)
	
iFileList = []

for fFileObj in os.walk(testDir + "/"):
	iFileList = fFileObj[2]
	break
print '%d test reviews.' % len(iFileList)

results = {"negative":0, "neutral":0, "positive":0}

print "\nFile Classifications:"

true_negative = 0
true_positive = 0
false_negative = 0
false_positive = 0
neg = 0
pos = 0


for filename in iFileList:

	fileText = bc.loadFile(testDir + filename)
	result = bc.classify(fileText)
    
        if "-1-" in filename or "-2-" in filename:
            neg = 1
            pos = 0
    
        elif "-5-" in filename or "-4-" in filename:
            neg = 0
            pos = 1
        else:
            neg = 0
            pos = 0

        if result is "positive":
            if pos == 1:
                true_positive += 1
            
            elif neg == 1:
                false_positive += 1

        elif result is "negative":
            if neg == 1:
                true_negative += 1
            
            elif pos == 1:
                false_negative += 1



        
	print "%s: %s" % (filename, result)
    
        if result in results:
            results[result] += 1
        else:
            results[result] = 1

print "\n"
print "# of true negatives: ", true_negative
print "# of false negatives: ",false_negative

print "# of true positives: ",true_positive
print "# of false positives: ",false_positive

accuracy = (float(true_positive) + float(true_negative)) / ( float(true_positive) + float(false_positive) + float(true_negative) + float(false_negative) )

precision_pos = float(true_positive) / (float(true_positive) + float(false_positive))
precision_neg = float(true_negative) / (float(true_negative) + float(false_negative))

recall_pos = float(true_positive) / (float(true_positive) + float(false_negative))
recall_neg = float(true_negative) / (float(true_negative) + float(false_positive))

fmeasure_pos = (2*precision_pos*recall_pos) / (precision_pos + recall_pos)
fmeasure_neg = (2*precision_neg*recall_neg) / (precision_neg + recall_neg)

print "accuracy is :" , accuracy
print "positive precision is :" , precision_pos
print "negative precision is :" , precision_neg
print "positive recall is :" , recall_pos
print "negative recall is :" , recall_neg
print "positive f-measure is :" , fmeasure_pos
print "negative f-measure is :" , fmeasure_neg

print "\nResults Summary:"
for r in results:
	print "%s: %d" % (r, results[r])
