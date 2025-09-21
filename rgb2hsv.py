import numpy as np
Inp=input("r,g,b").split(',')
r,g,b=map(int,Inp)
# r,g,b=r/255,g/255,b/255
Max=max([r,g,b])
Min=min([r,g,b])
mm=Max-Min
h=0
if Min==Max:
    h=0
elif Min==b:
    h=60*(g-r)/mm+60
elif Min==r:
    h=60*(b-g)/mm+180
elif Min==g:
    h=60*(r-b)/mm+300
s=mm
v=Max
print((h,s,v))