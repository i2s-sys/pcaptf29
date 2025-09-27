class Solution(object):
    def threeSum(self, nums):
        """ 先给数组从小到大排序，然后双指针 lo 和 hi 分别在数组开头和结尾
        和大一些，就让 lo++；和小一些，就让 hi--
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        return self.nSumTarget(nums, 3, 0, 0) # target为0 n为3 从nums[0]开始找
    def nSumTarget(self, nums, n, start, target) :
        sz = len(nums) - 1
        res = []
        low, high = start, sz
        if n == 2:
            while low < high:
                left, right = nums[low], nums[high]
                sum = left + right
                if sum > target:
                    while low < high and nums[high] == right:
                        high -= 1
                elif sum < target:
                    while low < high and nums[low] == left:
                        low += 1
                else:
                    res.append([left, right]) # 还不能退出 只是找到了其中的一对
                    while low < high and nums[high] == right:
                        high -= 1
                    while low < high and nums[low] == left:
                        low += 1
        else:
            for i in range(start, sz - 1):
                if i > start and nums[i] == nums[i - 1]:
                    continue
                sub = self.nSumTarget(nums, n - 1, start, target - nums[i])
                for arr in sub:
                    arr.append(nums[i])
                    res.append(arr)
        return res

if __name__ == '__main__':
    solution = Solution()
    input_strs = [-1,0,1,2,-1,-4]
    result = solution.threeSum(input_strs)
    print(result)
