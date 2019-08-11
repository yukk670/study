# 二分排序
nums = [12,10,1,3,5,9,11,6,2,4,8,7,13,14]
num = 15

def find(num,nums,left,right):
    if left > right:
        return None

    mid = (left+right) // 2
    if num < nums[mid]:
        return find(num,nums,left,mid-1)
    elif num > nums[mid]:
        return find(num,nums,mid+1,right)
    else:
        return mid
print( find(num,nums,0,len(nums) - 1) )

def fo(num,nums):
    left = 0
    right = len(nums) - 1
    while left <= right:

        mid = (left+right) // 2
        if num < nums[mid]:     right = mid - 1
        elif num > nums[mid]:   left = mid+1
        else:                   return mid

print(fo(num,nums))
# 冒泡排序
nums = [12, 10, 1, 3, 5, 9, 17, 6, 2, 4, 8, 7, 13, 14]
nums = [30, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
def a():
    count = 0;
    for i in range( len(nums) - 1):
        for j in range(i+1,len(nums)):
            count+=1
            if nums[i] > nums[j]:
                nums[i],nums[j] = nums[j],nums[i]

    print(nums,count)


def b():
    count = 0;
    for i in range(len(nums) - 1):
        result = True
        for j in range(len(nums)-i-1):
            count += 1
            a = nums
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
                result = False
        if result:
            break
    print(nums, count)

b()

# 快速排序
nums = [12, 10, 1, 3, 5, 9, 17, 6, 2, 4, 8, 7, 13, 14]
c = nums
# 无须列表
for i in range(1,len(nums)):
    # 取出当前无序数据和插入位置
    num,pos = nums[i],i
    # 有序列表
    for j in range(i-1,-1,-1):
        a = nums[j]
        b = nums[i]
        # 如果有序列表末尾值大于取出的无序列表的值
        if(nums[j] > num):
            # 记录可插入位置
            nums[j+1],pos = nums[j],j
        else:
            pos = j + 1
            break

    nums[pos] = num

print(nums)

# 插入排序
nums = [12, 10, 1, 3, 5, 9, 17, 6, 2, 4, 8, 7, 13, 14]

def quick(nums):
    if len(nums) >= 2:
        mark = nums[0]
        left,eq,right = [x for x in nums if x < mark],[x for x in nums if x == mark],[x for x in nums if x > mark]
        return quick(left) + eq + quick(right)
    else:
        return nums

print(quick(nums))