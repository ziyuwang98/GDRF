import torch

"""=============== BILINEAR ==============="""

bilinear_kernel = [None]
bilinear_kernel_4Channel = [None]

def bilinear_function(x):
    x = abs(x)
    res = 1 - x
    return res


def generate_bilinear_kernel(factor: int):
    size = 2 * factor
    func = torch.zeros(size, 1)
    for i in range(size):
        func[i] = bilinear_function((i - size / 2 + 0.5) / factor) / factor

    kernel = func @ func.t()

    return kernel


#------------------------------------ original one
def bilinear_kernel_init(num):
    global bilinear_kernel
    for i in range(1, num + 1):
        kernel2d = generate_bilinear_kernel(i)
        bilinear_kernel.append(torch.zeros(3, 3, 2 * i, 2 * i))
        bilinear_kernel[i][0, 0] = kernel2d
        bilinear_kernel[i][1, 1] = kernel2d
        bilinear_kernel[i][2, 2] = kernel2d

#------------------------------------ modified one
def bilinear_kernel_init_4Channel(num):
    global bilinear_kernel_4Channel
    for i in range(1, num + 1):
        kernel2d = generate_bilinear_kernel(i)
        bilinear_kernel_4Channel.append(torch.zeros(4, 4, 2 * i, 2 * i))
        bilinear_kernel_4Channel[i][0, 0] = kernel2d
        bilinear_kernel_4Channel[i][1, 1] = kernel2d
        bilinear_kernel_4Channel[i][2, 2] = kernel2d
        bilinear_kernel_4Channel[i][3, 3] = kernel2d

#---------------------------------------------------
def bilinear_downsample(input, factor: int):
    if input.shape[1] == 3:
        use_kernel = bilinear_kernel
    elif input.shape[1] == 4:
        use_kernel = bilinear_kernel_4Channel

    input = torch.nn.functional.pad(input, [int(0.5 * factor)] * 4, 'replicate')
    res = torch.nn.functional.conv2d(input, use_kernel[factor].to(input.device), stride=factor)

    return res

"""=============== BICUBIC ==============="""

bicubic_kernel = [None]
bicubic_kernel_4Channel = [None]

def bicubic_function(x):
    x = abs(x)
    if x <= 1:
        res = 1.5 * x**3 - 2.5 * x**2 + 1
    elif x < 2:
        res = -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2
    else:
        res = 0

    return res


def generate_bicubic_kernel(factor: int):
    size = 4 * factor
    func = torch.zeros(size, 1)
    for i in range(size):
        func[i] = bicubic_function((i - size / 2 + 0.5) / factor) / factor

    kernel = func @ func.t()

    return kernel

def bicubic_kernel_init(num):
    global bicubic_kernel
    for i in range(1,num + 1):
        kernel2d = generate_bicubic_kernel(i)
        bicubic_kernel.append(torch.zeros(3, 3, 4 * i, 4 * i))
        bicubic_kernel[i][0, 0] = kernel2d
        bicubic_kernel[i][1, 1] = kernel2d
        bicubic_kernel[i][2, 2] = kernel2d

def bicubic_kernel_init_4Channel(num):
    global bicubic_kernel_4Channel
    for i in range(1,num + 1):
        kernel2d = generate_bicubic_kernel(i)
        bicubic_kernel_4Channel.append(torch.zeros(4, 4, 4 * i, 4 * i))
        bicubic_kernel_4Channel[i][0, 0] = kernel2d
        bicubic_kernel_4Channel[i][1, 1] = kernel2d
        bicubic_kernel_4Channel[i][2, 2] = kernel2d
        bicubic_kernel_4Channel[i][3, 3] = kernel2d

def bicubic_downsample(input, factor:int):
    if input.shape[1] == 3:
        use_kernel = bicubic_kernel
    elif input.shape[1] == 4:
        use_kernel = bicubic_kernel_4Channel

    input = torch.nn.functional.pad(input, [int(1.5 * factor)] * 4, 'replicate')
    res = torch.nn.functional.conv2d(input, use_kernel[factor], stride=factor)
    
    return res


bilinear_kernel_init(16)
bicubic_kernel_init(16)

bilinear_kernel_init_4Channel(16)
bicubic_kernel_init_4Channel(16)