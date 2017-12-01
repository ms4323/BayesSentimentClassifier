import math, os, pickle, re
import numpy as np

class Bayes_Classifier:



   def __init__(self, trainDirectory = "movies_reviews/"):
      '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text.'''
   
      self.pos_dictionary = {}
      self.neg_dictionary = {}
      self.neg_document_len = 0
      self.pos_document_len = 0
   
      if os.path.isfile("pos_dict_best.txt") and os.path.isfile("neg_dict_best.txt"):
        print "pickled dictionaries found in directory, loading them into memory ..."
        self.pos_dictionary = self.load("pos_dict_best.txt")
        self.neg_dictionary = self.load("neg_dict_best.txt")
      else:
        print "pickled dictionaries not found in directory, training the system ..."
        self.train()


#res = self.classify("I love my AI class!")
#print res


   def train(self):   
      '''Trains the Naive Bayes Sentiment Classifier.'''
      training_path = "movies_reviews/"
      
      lFileList = []
      for fFileObj in os.walk(training_path):
        lFileList = fFileObj[2]
        break
   
      neg_document_len_list = [0]
      pos_document_len_list = [0]
      
      pos = 0
      neg = 0
      i =0
      while i < len(lFileList):
          
          fileString = self.loadFile(training_path + lFileList[i])
          tokenList = self.tokenize(fileString)
          
          if "-1-" in lFileList[i]:
            neg_document_len_list.append(len(tokenList))
            neg = 1
            pos = 0
          elif "-5-" in lFileList[i]:
            pos_document_len_list.append(len(tokenList))
            pos = 1
            neg = 0


          
          stopfileString = self.loadFile("stopwords.txt")
          stopwordsList = self.tokenize(stopfileString)
        
          
          j = 0
          while j < len(tokenList) :
              
            ## feature 1: each word added to dictionaries
            ## feature 2: ignoring the stop words in the text
            if tokenList[j] not in stopwordsList:
            
                word1 = tokenList[j]
                
                if neg == 1:
                    
                    if word1 is "not":
                        word2 = tokenList[j+1]
                        words = word1 + " " + word2 ## feature 3: two words together added to dictionaries
                        if words in self.neg_dictionary:
                            self.neg_dictionary[words] += 1
                    
                        else:
                            self.neg_dictionary[words] = 1
                    else:
                        if word1 in self.neg_dictionary:
                            self.neg_dictionary[word1] += 1

                        else:
                            self.neg_dictionary[word1] = 1
                    
                    
                elif pos == 1:
                    
                    if word1 is "not":
                        word2 = tokenList[j+1]
                        words = word1 + " " + word2 ## feature 3: two words together added to dictionaries
                        if words in self.pos_dictionary:
                            self.pos_dictionary[words] += 1
                        
                        else:
                            self.pos_dictionary[words] = 1
                    else:
                        if word1 in self.pos_dictionary:
                            self.pos_dictionary[word1] += 1
                                
                        else:
                            self.pos_dictionary[word1] = 1

            j+=1

          i+=1
              
      ##feature 4: length of document
      self.pos_document_len = np.median(pos_document_len_list) #median of lengths
      self.neg_document_len = np.median(neg_document_len_list) #median of lengths
      
      #saving the negative and positive dictionaries using pickles
      self.save(self.pos_dictionary,"pos_dict_best.txt")
      self.save(self.neg_dictionary,"neg_dict_best.txt")

   def classify(self, sText):
      '''Given a target string sText, this function returns the most likely document
      class to which the target string belongs. This function should return one of three
      strings: "positive", "negative" or "neutral".
      '''
      best_prop = None
      classifyAs = "undefine"
      
      neg_total_words = float(sum(self.neg_dictionary.values())) # total word count in neg
      
      pos_total_words = float(sum(self.pos_dictionary.values())) # total word count in pos
      
      neg_prop =  neg_total_words / (pos_total_words + neg_total_words) # probability of being in neg class
      
      pos_prop =  pos_total_words / (pos_total_words + neg_total_words) # probability of being in pos class
      
    
      sumword_prop_neg = 0.0 # sum of word probability for neg
      sumword_prop_pos = 0.0 # sum of word probability for pos
      
      distinct_words_count = float(len(self.neg_dictionary))
      for ww in self.pos_dictionary:
          if ww in self.neg_dictionary:
              distinct_words_count += 0
          else:
              distinct_words_count += 1
      
      document_len = 0
      for w in sText.split():
        document_len += 1
        if w in self.neg_dictionary:
            
            wordcount_in_neg = float(self.neg_dictionary[w]) # total word count for w in neg dictionary
        else:
            wordcount_in_neg = 0
        
        if w in self.pos_dictionary:
            wordcount_in_pos = float(self.pos_dictionary[w]) # total word count for w in pos dictionary
        else:
            wordcount_in_pos = 0

        wordcount_all = wordcount_in_pos + wordcount_in_neg

        ### Applying smoothing and underflow
        
        #adding one to the fraction when word not found in each dictionary
        if wordcount_all == 0:
            neg_res = 0 #log(1)
            pos_res = 0 #log(1)
        elif wordcount_in_neg == 0:
            neg_res = 0 #log(1)
            fraction_pos = (wordcount_in_pos / wordcount_all)
            pos_res = np.log( fraction_pos / pos_prop)
        elif wordcount_in_pos == 0:
            pos_res = 0 #log(1)
            fraction_neg = (wordcount_in_neg / wordcount_all)
            neg_res = np.log( fraction_neg / neg_prop)
        else:
            fraction_neg = (wordcount_in_neg / wordcount_all)
            fraction_pos = (wordcount_in_pos / wordcount_all)
            neg_res = np.log( fraction_neg / neg_prop)
            pos_res = np.log( fraction_pos / pos_prop)
        
        # sum all logs
        sumword_prop_neg += neg_res
        sumword_prop_pos += pos_res
       
      # add the sum with the log of class probalility
      result_neg = np.log(neg_prop) + sumword_prop_neg
      result_pos = np.log(pos_prop) + sumword_prop_pos


      if result_neg - result_pos > 0.5 :
        best_prop = result_neg
            #if self.neg_document_len - document_len > 1:
        classifyAs = "negative"
            #else:
            #classifyAs = "neutral"
      elif result_pos - result_neg > 0.5 :
        best_prop = result_pos
            #if document_len - self.pos_document_len > 1:
        classifyAs = "positive"
            #else:
            #classifyAs = "neutral"

      else:
        classifyAs = "neutral"

      return classifyAs
          
          
   def loadFile(self, sFilename):
      '''Given a file name, return the contents of the file as a string.'''

      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      '''Given an object and a file name, write the object to the file using pickle.'''

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      '''Given a file name, load and return the object stored in the file.'''

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      '''Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order).'''

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens


#BC = Bayes_Classifier()

