with open('abalone.txt') as f:
    lines = f.readlines()

train_lines = lines[:3130]
test_lines = lines[3130:]

with open('abalone.train', 'w') as train_f:
    train_f.writelines(train_lines)

with open('abalone.test', 'w') as test_f:
    test_f.writelines(test_lines)
