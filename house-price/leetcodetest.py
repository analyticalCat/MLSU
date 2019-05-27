import numpy as np

def twoSums(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """

    """     1236 ms
    for a in nums:
        b = target - a
        
        if b != a: 
            try:
                return [nums.index(a), nums.index(b)]
            except:
                pass
        else:
            try:
                c = [i for i, x in enumerate(nums) if x == b][1]
                return [nums.index(a), c]
            except:
                pass 

    for a in nums:   #1168ms
        b = target - a
        
        if b in nums: 
            if b != a:
                return [nums.index(a), nums.index(b)]
            else:
                try:
                    c = [i for i, x in enumerate(nums) if x == b][1]
                    return [nums.index(a), c]
                except:
                    pass  
    
  """
    #use dictionary to store key value pair to achieve best performance on search
    pairs = {}  #36ms
    for i, a in enumerate(nums):   #
        b = target - a

        if a in pairs:
            return [pairs[a], i]
        else:
            pairs[b] = i
  ####################################################################      
def countandsay(n):
    if n > 30: return None
    
    #return n's term of count and say
    #first, use list and recursion.  no data structure optimization
    if n==1: 
        return "1"
    else:
        return countingLast(countandsay(n-1))


def countingLast(lastSay):
    if len(lastSay)==0: return None
    ret = ""
    xLast = 0
    i = 0

    for x in lastSay:
        if i == 0: 
            xLast = x
            xCount = 1
        else:
            if xLast == x:
                xCount += 1
            else:
                ret = ret + str(xCount)
                ret = ret +str(xLast)     
                xLast = x
                xCount = 1 
        i += 1
      
    
    ret = ret + str(xCount)
    ret = ret + str(xLast)
    return ret

#print(countandsay(5))

####################################################################


def generateTimesmatrix(M):
    #Initialize the array with 1s.  Just for ease of calculation
    import random

    times =  []
    i = 0
    while i<M:
        j = 0
        while j < M:
            if i == j:
                if i == 0:
                    times = np.array([0,0,0])
                else:
                    times = np.append(times, [[i,j,0]], axis = 0)
            elif i == 0 and j == 1:
                times = np.append([times], [[i,j,random.randint(1,100)]], axis = 0)
            else:
                times = np.append(times, [[i,j,random.randint(1,100)]], axis = 0)
            j +=1
        i += 1

    return times            

def networkDelayTime(times, N, K) ->int:
    #N nodes in total.  
    #signal starting from K.  
    # How long does it take for all nodes to receive signal.  return -1 when impossible

    #use dictionary to store data, tuples of nodes to be the key
    pairs = {}
    maxtime = 0
    
    for x in times:
       pairs[(x[0], x[1])] = x[2]

    for i in range(0,N-1):
        #find the time from k to i, if can't find one, return -1
        if (K,i) in pairs.keys():
            nodetime = pairs[(K,i)]      
            if nodetime == -1: 
                return -1
            #record the max 
            maxtime = max(maxtime, nodetime)
    return maxtime

    

N=5
K=1
timesM = generateTimesmatrix(N)
print(networkDelayTime(timesM, 5,1))
#delayTime = networkDelayTime(timesM, N, K)
