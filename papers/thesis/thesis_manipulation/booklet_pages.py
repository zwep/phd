import itertools
n = 12
range_1 = list(range(n-1, 0, -1))
range_2 = list(range(1, n))
z = [n]
counter = 0
while len(range_2) and (len(range_1)):
    print(counter)
    counter += 1
    if counter <= 2:
        temp = range_1.pop()
        print('\t',temp)
        z.append(temp)
    else:
        temp = range_2.pop()
        z.append(temp)
        if len(range_2):
            temp = range_2.pop()
            z.append(temp)
        counter = 0

','.join([str(x+50) for x in z])