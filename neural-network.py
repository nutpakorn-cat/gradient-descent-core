import random
import numpy

# setting
ENABLE_LOG = False

# initial variables
nn_num = 0
nn_list = []

input_num = 0
input_list = []

w_num = 0
w_list = []

b_num = 0
b_list = []

target_num = 0
target_list = []

gradient_list_b = []
gradient_list_w = []

outdate = False

def check_outdate():
    if(outdate):
        print('Warning: nn_list is outdated.')

def sigmoid(x):
    """Calculate a value of sigmoid for input x

    Keyword arguments:
    x -- the input
    """
    return 1/(1 + numpy.exp(-x))

def nn_init(num_of_input, num_of_output):
    """Initialize neural network.

    Keyword arguments:
    num_of_input -- the number of input 
    num_if_output -- the number of output
    """
    global input_num
    global nn_num
    global w_num
    global b_num
    global target_num
    
    global nn_list
    global input_list
    global w_list
    global b_list
    global target_list
    global gradient_list_b
    global gradient_list_w
    
    input_num = num_of_input
    nn_num = target_num = num_of_output
    
    for i in range(nn_num):
        nn_list.append(0)
        target_list.append(0)
    if(ENABLE_LOG): print(nn_list)
        
    for i in range(input_num):
        input_list.append(0)
    if(ENABLE_LOG): print(input_list)
    
    w_num = input_num * nn_num
    
    for i in range(w_num):
        w_list.append(random.randint(-10, 10))
    if(ENABLE_LOG): print(w_list)
    
    b_num = nn_num
    
    for i in range(b_num):
        b_list.append(random.randint(-10, 10))
    if(ENABLE_LOG): print(b_list)

    gradient_list_w = [0] * (nn_num*input_num)
    gradient_list_b = [0] * nn_num

def nn_calculate(nn_index):   
    """Calculate an output node particularly.

    Keyword arguments:
    nn_index -- the index of output node.
    """
    weight_start_index = input_num * nn_index
    
    global outdate
    
    sum_of_wa = 0
    
    # calculate sum of wi * ai 
    for i in xrange(weight_start_index, 
                    (weight_start_index + input_num)):
        sum_of_wa += w_list[i] * input_list[i - weight_start_index]
        
        if(ENABLE_LOG): print('for index: ' 
                             + str(nn_index) 
                             + ' i:' + str(i) 
                             + ' sum_of_wa: '
                             + str(sum_of_wa))
    z = sum_of_wa + b_list[nn_index]
    outdate = False
    
    if(ENABLE_LOG):
        print('z(' + str(nn_index) + ') = sigmoid(' 
          + str(sum_of_wa) 
          + ' + ' 
          + str(b_list[nn_index])
          + ')'
          + ' = '
          + str(sigmoid(z)))
        
    # calculate sum of 'sum_of_wa' and b
    
    return sigmoid(z)

def nn_calculate_all():
    global outdate
    global nn_list
    
    for i in xrange(nn_num):
        nn_list[i] = nn_calculate(i)
    outdate = False

def input_update(index, value):
    """Update input value.

    Keyword arguments:
    index -- the index of input list
    value -- the value for updating value
    """
    global input_list
    global outdate
    
    input_list[index] = value
    outdate = True
    
def w_update(index, value):
    """Update w value.

    Keyword arguments:
    index -- the index of input list
    value -- the value for updating value
    """
    global w_list
    global outdate
    
    w_list[index] = value
    outdate = True

def b_update(index, value):
    """Update b value.

    Keyword arguments:
    index -- the index of input list
    value -- the value for updating value
    """
    global b_list
    global outdate
    
    b_list[index] = value
    outdate = True

def target_update(index, value):
    """Update target value.

    Keyword arguments:
    index -- the index of target list
    value -- the value for updating value
    """
    global target_list
    global outdate
    
    target_list[index] = value
    outdate = True
    
def cost_calculate():
    """Calculate cost value."""
    if(ENABLE_LOG): print('COST CALCULATE' )
    
    global nn_list
    global outdate
    
    cost = 0
    
    for i in xrange(nn_num):
        nn_list[i] = nn_calculate(i)
        
        if(ENABLE_LOG): print('(' 
                              + str(nn_list[i]) 
                              + ' - '
                              + str(target_list[i]) 
                              +')^2 = ' 
                              + str((nn_list[i]-target_list[i]) ** 2))
        cost += (nn_list[i]-target_list[i]) ** 2
    
    outdate = False
    return cost

def cost_calculate_at(index):
    """Calculate cost value particularly."""
    global nn_list
    global outdate

    i = index
    nn_list[i] = nn_calculate(i)
        
    if(ENABLE_LOG): print('(' 
                            + str(nn_list[i]) 
                            + ' - '
                            + str(target_list[i]) 
                            +')^2 = ' 
                            + str((nn_list[i]-target_list[i]) ** 2))
    outdate = False
    
    return (nn_list[i]-target_list[i]) ** 2

def slope_w_at(m):
    k = (m)%input_num
    i = (m - k) / input_num
    #print('Check slope w at ' + str(m) + 'is slope for input: ' + str(k))
    #print('And slope for output: ' + str(i))
    check_outdate()
    
    epsilon = nn_list[i] * (1-nn_list[i])
    u = nn_list[i] - target_list[i]
    
    return 2*u * epsilon * input_list[k]

def slope_b_at(i):
    #print('Check slope b at ' + str(i) + 'slope for output ' + str(i))
    check_outdate()
    
    epsilon = nn_list[i] * (1-nn_list[i])
    u = nn_list[i] - target_list[i]
    
    return 2*u * epsilon

def train(time, learning_rate):
    global gradient_list_w
    global gradient_list_b

    before_train = cost_calculate()
    gradient_before_w = [0] * (nn_num*input_num)
    gradient_before_b = [0] * nn_num

    print('TRAINING SESSION')
    print('BEFORE TRAIN -> cost: ' + str(before_train) + 'target: ' + str(target_list[0]) + ' , ' + str(target_list[1]) )
    print('OUTPUT VALUE: ' + str(nn_list[0]) + ' , ' + str(nn_list[1]))
    print('START!!!')
    print('================================================')
    
    for t in xrange(time):
        for i in xrange(nn_num):
            for k in xrange(input_num):
                if(t == False):
                    gradient_before_w[i+k] = w_list[i+k]
                    gradient_before_b[i] = b_list[i]
                w_update(i+k, w_list[i+k] + (-learning_rate * slope_w_at(i+k)))
                nn_calculate_all()
            b_update(i,b_list[i] + (-learning_rate * slope_b_at(i)))
            nn_calculate_all()
        print('Train at: ' + str(t) + ' cost: ' + str(cost_calculate()))
    
    after_train = cost_calculate()
    
    print('FINISHED!!!')

    print('AFTER TRAIN -> cost: ' 
    + str(after_train) 
    + 'target: ' 
    + str(target_list[0]) 
    + ' , ' 
    + str(target_list[1]) )

    print('OUTPUT VALUE: ' 
    + str(nn_list[0]) 
    + ' , ' 
    + str(nn_list[1]))

    print('IMPROVEMENT SCORE: ' 
    + str(before_train - after_train))

    for i in xrange(nn_num):
        for k in xrange(input_num):
            gradient_list_w[i+k] += w_list[i+k] - gradient_before_w[i+k]

            print('W m: '
            + str(i+k) 
            + ' = '
            + str(w_list[i+k] 
            - gradient_before_w[i+k]))
        gradient_list_b[i] += b_list[i] - gradient_before_b[i]

        print('B i: '
        + str(i) 
        + ' = '
        + str(b_list[i] - gradient_before_w[i]))
    
# Coding Section

nn_init(2, 2)
input_update(0, 0.5)
input_update(1, 0.3)

print(input_list)

print(target_list)
target_update(0, 1)
target_update(1, 0.24)
print(target_list)

print(nn_calculate_all())
train(100000, 0.01)


        