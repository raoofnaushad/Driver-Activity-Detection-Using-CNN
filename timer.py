import threading 
# i = 0
def gfg():

    # timer = threading.Timer(5.0, gfg) 
    # timer.start() 
    # # print("Cancelling timer\n") 
    # # timer.cancel() 
    print("Exit\n") 
    print("GeeksforGeeks\n")


timer = threading.Timer(5.0, gfg )
timer.start() 
# print("Cancelling timer\n") 
# timer.cancel() 
print("Exit\n") 

  
 

