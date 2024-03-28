import random
import sympy as sp
import time
import multiprocessing as mp
from gmpy2 import invert


# 多线程回调函数
def call_back(res):
    print(f'result: {res}')


# 多线程错误回调函数
def err_call_back(err):
    print(f'error: {str(err)}')


def is_primitive_root(g, p, factors):
    # determine whether g is a primitive root of p
    for factor in factors:
        if pow(g, (p-1)//factor, p) == 1:
            return False
    return True


def generate_p_and_g(n_bit):
    while True:
        # generate an n-bit random prime number p
        p = sp.randprime(2**(n_bit-1), 2**n_bit)

        # compute the prime factorization of p-1
        factors = sp.factorint(p-1).keys()

        # choose a possible primitive root g
        for g in range(2, p):
            if is_primitive_root(g, p, factors):
                return p, g


def mod_exp(base, exponent, modulus):
    """TODO: calculate (base^exponent) mod modulus. 
        Recommend to use the fast power algorithm.
    """
    if base == 0:
        return 0

    if sp.isprime(modulus):
        # 费马小定理加速快速幂
        exponent = exponent % (modulus - 1)

    # 快速幂算法迭代版本
    result = 1
    base = base % modulus

    while exponent > 0:
        if exponent & 1:
            result = (result * base) % modulus
        exponent >>= 1
        base = (base * base) % modulus

    return result


def elgamal_key_generation(key_size):
    """Generate the keys based on the key_size.
    """
    # generate a large prime number p and a primitive root g
    p, g = generate_p_and_g(key_size)

    # TODO: generate x and y here.
    # 生成私钥 0<x<p-1
    x = random.randint(1, p-2)
    # 生成公钥 y = g^x mod p
    y = mod_exp(g, x, p)

    return (p, g, y), x


def elgamal_encrypt(public_key, plaintext):
    """TODO: encrypt the plaintext with the public key.
    """
    # 公钥
    p, g, y = public_key
    # 生成临时私钥 1<k<p-1
    k = random.randint(1, p-2)
    # 计算临时公钥 c1 = g^k mod p 和临时密文 c2 = m * y^k mod p
    c1, c2 = mod_exp(g, k, p), (plaintext * mod_exp(y, k, p)) % p
    return c1, c2


def elgamal_decrypt(public_key, private_key, ciphertext):
    """TODO: decrypt the ciphertext with the public key and the private key.
    """
    p, g, y = public_key
    c1, c2 = ciphertext
    # 利用私钥计算临时公钥的模反演
    s = mod_exp(c1, private_key, p)
    # 计算明文消息
    plaintext = (c2 * invert(s, p)) % p
    return plaintext


def mul_elgamal_encrypt_decrypt(nums_begin, nums_end, public_key, private_key):
    """TODO: encrypt the plaintext with the public key.
    """
    # 原地加密原地解密，也可输出文件暂未实现
    for i in range(nums_begin, nums_end, 10000):
        ciphertext = elgamal_encrypt(public_key, i)
        elgamal_decrypt(public_key, private_key, ciphertext)


if __name__ == "__main__":

    test = int(
        input("请输入要测试的项目：[1] 随机性质 [2]乘法同态性质 [3] 不同 key_size 下三个阶段的时间开销 [4] 大数据测试\n"))
    if test == 2:
        # 验证乘法同态性质
        print("Test_mul:")
        # set key_size, such as 256, 1024...
        key_size = int(input("Please input the key size: "))

        # generate keys
        public_key, private_key = elgamal_key_generation(key_size)
        print("Public Key:", public_key)
        print("Private Key:", private_key)

        time_no_use = []
        time_use_mul = []
        for i in range(100):
            plaintext1 = random.randint(1, public_key[0]-2)
            print("Plaintext1:", plaintext1)
            ciphertext1 = elgamal_encrypt(public_key, plaintext1)
            print("Ciphertext1:", ciphertext1)
            plaintext2 = random.randint(1, public_key[0]-2)
            print("Plaintext2:", plaintext2)
            ciphertext2 = elgamal_encrypt(public_key, plaintext2)
            print("Ciphertext2:", ciphertext2)
            time_start = time.time()
            decrypted_text1 = elgamal_decrypt(
                public_key, private_key, ciphertext1)
            print("Decrypted Text1:", decrypted_text1)
            decrypted_text2 = elgamal_decrypt(
                public_key, private_key, ciphertext2)
            print("Decrypted Text2:", decrypted_text2)
            time_end = time.time()
            time_no_use.append(time_end-time_start)
            print('time cost', time_end-time_start, 's')
            ciphertext1 = elgamal_encrypt(public_key, plaintext1)
            print("Ciphertext1:", ciphertext1)
            ciphertext2 = elgamal_encrypt(public_key, plaintext2)
            print("Ciphertext2:", ciphertext2)
            mul_ciphertext = (
                ciphertext1[0] * ciphertext2[0], ciphertext1[1] * ciphertext2[1])
            time_start = time.time()
            decrypted_text = elgamal_decrypt(
                public_key, private_key, mul_ciphertext)
            print("Decrypted Text Mul:", decrypted_text)
            time_end = time.time()
            time_use_mul.append(time_end-time_start)
            print('time cost', time_end-time_start, 's')
            if decrypted_text == plaintext1 * plaintext2:
                print("乘法同态性质成立！")
            else:
                print("乘法同态性质不成立！")

        print('plain average time cost', sum(
            time_no_use)/len(time_no_use), 's')
        print('Multiplicative homomorphic properties average time cost',
              sum(time_use_mul)/len(time_use_mul), 's')
    elif test == 1:
        # 验证随机性质
        # set key_size, such as 256, 1024...
        key_size = int(input("Please input the key size: "))

        # generate keys
        public_key, private_key = elgamal_key_generation(key_size)
        print("Public Key:", public_key)
        print("Private Key:", private_key)

        print("Test_random:")
        plaintext = random.randint(1, public_key[0]-2)
        ciphertext = set()
        for i in range(5):
            temp = elgamal_encrypt(public_key, plaintext)
            print("Ciphertext:", temp)
            if temp in ciphertext:
                print("随机性质不成立！")
                break
        print("随机性质成立！")
    elif test == 3:
        # 测试不同 key_size 设置下三个阶段的时间开销
        print("Test_key_size:")
        time_key_generation = []
        time_encryption = []
        time_decryption = []
        for key_size in [64, 128, 256]:
            time_start = time.time()
            public_key, private_key = elgamal_key_generation(key_size)
            time_end = time.time()
            time_key_generation.append(time_end-time_start)
            print('key_size', key_size, 'generation time cost',
                  time_end-time_start, 's')

            plaintext = []
            ciphertext = []
            time_start = time.time()
            for i in range(50):
                plaintext.append(random.randint(1, public_key[0]-2))
                ciphertext.append(elgamal_encrypt(public_key, plaintext[i]))
            time_end = time.time()
            time_encryption.append(time_end-time_start)
            print('key_size', key_size, 'encrypt time cost',
                  time_end-time_start, 's')

            decrypted_text = []
            time_start = time.time()
            for i in range(50):
                decrypted_text.append(elgamal_decrypt(
                    public_key, private_key, ciphertext[i]))
            time_end = time.time()
            time_decryption.append(time_end-time_start)
            print('key_size', key_size, 'decryption time cost',
                  time_end-time_start, 's')
    elif test == 4:
        print("Test_big_data:")
        # set key_size, such as 256, 1024...
        key_size = int(input("Please input the key size: "))

        # generate keys
        public_key, private_key = elgamal_key_generation(key_size)
        print("Public Key:", public_key)
        print("Private Key:", private_key)

        print("预计算实现")
        time_start = time.time()
        # 预计算 g^i mod p 和 y^i mod p
        g = []
        for i in range(1, 1000):
            g.append(mod_exp(public_key[1], i, public_key[0]))

        y = []
        for i in range(1, 1000):
            y.append(mod_exp(public_key[2], i, public_key[0]))

        # 预计算 i^x mod p
        c1_x = []
        for i in range(1000):
            c1_x.append(mod_exp(i, private_key, public_key[0]))

        # 生成密文
        encrypted_text = []
        for i in range(1, int(1e9), 10000):
            k = random.randint(1, public_key[0]-2)
            if k < 1000:
                # 计算临时公钥 c1 = g^k mod p 和临时密文 c2 = m * y^k mod p
                ciphertext = g[k-1], (i * y[k-1]) % public_key[0]
            else:
                ciphertext = mod_exp(public_key[1], k, public_key[0]), (
                    i * mod_exp(public_key[2], k, public_key[0])) % public_key[0]
            encrypted_text.extend(ciphertext)
        # 解密
        decrypted_text = []
        for i in range(0, len(encrypted_text), 2):
            ciphertext = encrypted_text[i:i+2]
            if ciphertext[0] < 1000:
                # 利用私钥计算临时公钥的模反演
                s = c1_x[ciphertext[0]]
                # 计算明文消息
                decrypted_text.append(
                    (ciphertext[1] * invert(s, public_key[0])) % public_key[0])
            else:
                decrypted_text.append(elgamal_decrypt(
                    public_key, private_key, ciphertext))
        time_end = time.time()
        time1 = time_end-time_start

        print("多线程实现")
        time_start = time.time()
        # 加密解密
        # 创建线程池
        pool = mp.Pool(processes=mp.cpu_count())
        # 每个线程分配百万个数，内部步长为一万
        for i in range(1, int(1e9), 1000000):
            pool.apply_async(mul_elgamal_encrypt_decrypt, args=(
                i, i+1000000, public_key, private_key))
        pool.close()
        pool.join()
        time_end = time.time()
        time2 = time_end-time_start

        print("朴素实现")
        time_start = time.time()
        # 生成密文
        encrypted_text = []
        for i in range(1, int(1e9), 10000):
            ciphertext = elgamal_encrypt(public_key, i)
            encrypted_text.extend(ciphertext)
        # 解密
        decrypted_text = []
        for i in range(0, len(encrypted_text), 2):
            decrypted_text.append(elgamal_decrypt(
                public_key, private_key, encrypted_text[i:i+2]))
        time_end = time.time()
        time3 = time_end-time_start
        print('预计算 time cost', time1, 's')
        print('多线程 time cost', time2, 's')
        print('朴素 time cost', time3, 's')
    else:
        print("输入错误！")
