class Solution:
    def search(self, nums: list[int], target: int) -> int:
        l = 0
        h = len(nums)-1
        mid = 0
        while l<h:
            mid = round((l+h)/2)
            print("l ", l)
            print("h ", h)
            print("mid ", mid)
            if nums[mid] == target:
                print("in if")
                return mid
            elif nums[mid] > target:
                # print("elif")
                h = mid + 1
            else:
                # print("else")
                l = mid - 1

        return -1

obj = Solution()
print(obj.search([1,2,3,4,5,6], 5))

