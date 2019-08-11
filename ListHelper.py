# class ListHelper:
#     @staticmethod
#     def count(list_target,func_condition):
#         """
#         根据条件返回列表内所有符合条件的元素数量
#         :param list_target: 指定列表
#         :param func_condition: 条件
#         :return: 符合条件的元素数量
#         """
#         return len([item for item in list_target if func_condition(item)])
#
#     @staticmethod
#     def first(list_target, func_condition):
#         """
#         根据条件返回列表内符合条件的一个元素
#         :param list_target: 指定列表
#         :param func_condition: 条件
#         :return: None未查询到
#         """
#         for item in list_target:
#             if func_condition(item):
#
#                 return item
#
#     @staticmethod
#     def all(list_target, func_condition):
#         """
#         根据条件返回列表内所有符合条件的元素列表
#         :param list_target: 指定列表
#         :param func_condition: 条件
#         :return: None未查询到
#         """
#         return (item for item in list_target if func_condition(item))
#
#     @staticmethod
#     def max(list_target,func_condition):
#         max = 0
#         for item in list_target:
#             if max < func_condition(item):
#                 max = func_condition(item)
#         return max
#
#     @staticmethod
#     def sum(list_target, func_condition):
#         """
#         根据条件返回列表内所有指定属性的和
#         :param list_target: 指定列表
#         :param func_condition: 对应属性
#         :return: 列表内所有指定属性的和
#         """
#         sum = 0
#         for item in list_target:
#             sum += func_condition(item)
#         return sum
#
#     @staticmethod
#     def get_property(list_target, func_condition):
#         """
#         根据条件返回列表内所有指定属性列表
#         :param list_target: 指定列表
#         :param func_condition: 对应属性
#         :return: 列表内所有指定属性的列表
#         """
#         return (func_condition(item) for item in list_target)
#
#     @staticmethod
#     def delete(list_target, func_condition):
#         """
#         根据条件删除列表内所有指定属性列表
#         :param list_target: 指定列表
#         :param func_condition: 对应属性
#         :return:
#         """
#         for i in range(len(list_target)-1,-1,-1):
#             if func_condition(list_target[i]):
#                 list_target.remove(list_target[i])