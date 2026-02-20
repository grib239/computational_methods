import numpy as np
from PIL import Image
import struct
import numpy.typing as npt
import scipy.linalg
import os



def read_bmp(path):
    img = Image.open(path).convert('L')
    image_array = np.array(img, dtype=np.float32)
    return image_array

def read_bmp_in64(path):
    img = Image.open(path).convert('L')
    image_array = np.array(img, dtype=np.float64)
    return image_array


def save_bmp(matrix, output_path):
    img = Image.fromarray(matrix.astype(np.uint8))
    img.save(output_path)


def compute_svd_numpy(matrix, k=0):
    return np.linalg.svd(matrix, full_matrices=False)

def compute_svd_scipy(matrix, k=0):
    return scipy.linalg.svd(matrix, lapack_driver='gesvd')


def compute_compression(matrix, compression_level = 2, svd_computer=compute_svd_numpy):
    rows, cols = matrix.shape
    original_size = rows * cols 

    rank = max(1, min(rows, cols, original_size // (compression_level * (rows + cols + 1)*4)))

    u, s, vt = svd_computer(matrix, rank)
    
    if rank*(rows + cols + 1) * 4 > original_size//compression_level:
        print("compression level is too high")
        return None
    u = u[:, :rank]
    s = s[:rank]
    vt = vt[:rank, :]

    return u, s, vt, rank


def save_custom(U, S, Vt, original_shape, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        f.write(struct.pack('III', *original_shape, U.shape[1]))
        f.write(S.astype(np.float32).tobytes())
        f.write(U.astype(np.float32).tobytes())
        f.write(Vt.astype(np.float32).tobytes())


def load_custom(filename):
    with open(filename, 'rb') as f:
        rows, cols, rank = struct.unpack('III', f.read(12))
        S = np.frombuffer(f.read(rank * 4), dtype=np.float32)
        U = np.frombuffer(f.read(rows * rank * 4), dtype=np.float32).reshape(rows, rank)
        Vt = np.frombuffer(f.read(rank * cols * 4), dtype=np.float32).reshape(rank, cols)
    return U, S, Vt, (rows, cols)


def compress_image(path_to_bmp, output_custom, compression_level = 2):
    matrix = read_bmp(path_to_bmp)
    U, S, Vt, rank = compute_compression(matrix, compression_level, randomazid_svd)

    save_custom(U, S, Vt, matrix.shape, output_custom)
    
    print(f"Compressed: rank={rank}, compression factor≥{compression_level}")
    return rank

def decompress_image(path_custom, output_bmp):
    U, S, Vt, shape = load_custom(path_custom)
    reconstructed = (U @ np.diag(S) @ Vt).clip(0, 255)
    
    save_bmp(reconstructed, output_bmp)
    print(f"Decompressed: shape={shape}")


def calculate_polynom(x, y, c=[0]*10):
    return (c[0] + c[1]*x + c[2]*y+ c[3]*(x**2) + c[4]*(y**2) + c[5]*(x*y) + c[6]*(x**3) + c[7]*(y**3) + c[8]*(x*x*y) + c[9]*(x*y*y)) % 256

def create_matrix(m=800, n=533, coef=[3,-8,4,-1,-10,-10,-1,8,4,-5]):
    M = np.zeros((m, n))
    M = np.fromfunction(lambda i, j: calculate_polynom(i, j, c=coef), (m, n))
    return M

def one_step_opt():
    c = np.random.randint(low=-10, high=10, size=10)
    M = create_matrix(coef=c)
    diff = calculate_difference_metric(M)
    return diff, c


def calculate_difference_metric(M):
    s1 = compute_svd_numpy(M)[1]
    s2 = compute_svd_scipy(M)[1]

    a = np.sort(s1)
    b = np.sort(s2)

    mask = (a != 0) & (b != 0)
    c = np.zeros_like(a)
    c[mask] = np.maximum(a[mask]/b[mask], b[mask]/a[mask])

    return np.linalg.norm(c)


def randomazid_svd(M, k=30):
    m, n = M.shape
    Omega = np.random.uniform(size=(n, k))
    Y = M @ Omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ M
    u, S, V = np.linalg.svd(B, full_matrices=0)
    U = Q @ u
    return U, S, V


def PSNR(M, N):
    assert N.shape == M.shape
    MSE = ((M - N) ** 2).mean()
    return 10 * np.log10((255**2) / MSE)


def SSIM(M, N):
    assert N.shape == M.shape
    dN = (N ** 2 - N.mean() ** 2).mean()
    dM = (M ** 2 - M.mean() ** 2).mean()
    cov = ((N - N.mean())*(M - M.mean())).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return (2*M.mean() * N.mean() + c1)*(2 * cov + c2) / (N.mean() ** 2 + M.mean() ** 2 + c1) / (dM + dN + c2)

"""
input_files = ["pigs/pig1.bmp", "pigs/pig2.bmp", "pigs/pig3.bmp"]
comp_values = [2, 4, 8]
for compression_level in comp_values:
    for i in range(len(input_files)):
        file = input_files[i]

        compress_image(file, f"compressed_random_svd{compression_level}_{file[:-4]}", compression_level)
        decompress_image(f"compressed_random_svd{compression_level}_{file[:-4]}", (f"compressed_random_svd{compression_level}_{file}"))
"""
for i in [2, 4, 8]:
    folder1 = f"compressed_random_svd{i}_pigs"
    folder2 = f"compressed_{i}_pigs"
    for j in range(1, 4, 1):
        A = read_bmp(f"pigs/pig{j}.bmp")
        B = read_bmp(f"{folder1}/pig{j}.bmp")
        C = read_bmp(f"{folder2}/pig{j}.bmp")
        print(f"======pig:{j}___comp_level:{i}=======")
        print("=====PSNR=====")
        print(f"randomized: {PSNR(A, B)}")
        print(f"standard  : {PSNR(A, C)}")
        print("=====SSIM======")
        print(f"randomized: {SSIM(A, B)}")
        print(f"standard  : {SSIM(A, C)}")


"""
M = read_bmp_in64("kill_image.bmp")

dif = calculate_difference_metric(M)
print(f"difference calculate using L2(max(a/b, b/a)): {dif}")

best_dif = 0
best_c = [0,0,0,0,0,0,0,0,0,0]
for i in range(100):
    dif, c = one_step_opt()
    if dif > best_dif:
        print(dif)
        best_dif = dif
        best_c = c
print(best_dif)
print(best_c)

"""