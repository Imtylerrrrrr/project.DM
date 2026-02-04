arr = [] #배열 선언
k=0
for i in range(19):

    arr.append([])
    arr[i] = list(map(int, input().split())) #배열에 넣기

a= int(input())
for i in range(a):
    x,y = input().split()
    for j in range(19):
        if j != int(y):
            if arr[int(x)][j] == 0:
                arr[int(x)][j] = 1
            else:
                arr[int(x)][j] = 0

    for j in range(19):
        if j != int(x):
            if arr[j][int(y)] == 0:
                arr[j][int(y)] = 1
            else:
                arr[j][int(y)] = 0 #뒤집기

for i in arr :
    for j in i:
        print(j,end=" ")
    print()