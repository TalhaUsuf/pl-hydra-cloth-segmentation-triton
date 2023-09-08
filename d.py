

def reverse(_input : str):
	if isinstance(_input, str):
		_out = []
		idx = 0
		for k in range(len(list(_input)), 0, -1):
			# k will go from 3 ---> 0
			_out[idx] = list(_input)[k]
			idx += 1

		_out = ''.join(_out)

		return _out
	else:
		raise ValueError(f'input {_input } must be a str instance')


# test the function

_input = '123'
out = reverse(_input)
assert out == '321'










Given a list, find the index of the second largest element of the list.
e.g. list = [5, 7, 1, 9, 2] solution = 1


# [5, 7, 1, 9, 2] --> sorted --> [9, 7, 5, 2, 1] --> idx = 1


# sort the array

input_ = [5, 7, 1, 9, 2]


min_current = -inf
idx= 0
flag = True

def sort(input_):
	sorted = []
	for i in range(1,len(input_), i):
		if input_[i] < input_[i-1]:
			min_current = input_[i-1]
			sorted.append(input_[i])				
		else:
			continue

	return sorted


while flag:

	out = sort(input_)
	if out 


























