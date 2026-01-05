import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage
from scipy.fft import dctn, idctn
from heapq import heappush, heappop
from collections import Counter

X = misc.ascent()
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

Y1 = dctn(X, type=1)
Y2 = dctn(X, type=2)
Y3 = dctn(X, type=3)
Y4 = dctn(X, type=4)
freq_db_1 = 20*np.log10(abs(Y1))
freq_db_2 = 20*np.log10(abs(Y2))
freq_db_3 = 20*np.log10(abs(Y3))
freq_db_4 = 20*np.log10(abs(Y4))

plt.subplot(221).imshow(freq_db_1)
plt.subplot(222).imshow(freq_db_2)
plt.subplot(223).imshow(freq_db_3)
plt.subplot(224).imshow(freq_db_4)
plt.show()


# se elimina frecventele inalte (cele care conribuie mai putin la perceptia vizuala)
k = 120

Y_ziped = Y2.copy()
Y_ziped[k:] = 0
X_ziped = idctn(Y_ziped)

plt.imshow(X_ziped, cmap=plt.cm.gray)
plt.show()

Q_down = 10

X_jpeg = X.copy()
X_jpeg = Q_down*np.round(X_jpeg/Q_down)

plt.subplot(121).imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(X_jpeg, cmap=plt.cm.gray)
plt.title('Down-sampled')
plt.show()

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

# Encoding
x = X[:8, :8]
y = dctn(x)
y_jpeg = Q_jpeg*np.round(y/Q_jpeg)

# Decoding
x_jpeg = idctn(y_jpeg)

# Results
y_nnz = np.count_nonzero(y)
y_jpeg_nnz = np.count_nonzero(y_jpeg)

plt.subplot(121).imshow(x, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(x_jpeg, cmap=plt.cm.gray)
plt.title('JPEG')
plt.show()

print('Componente in frecventa:' + str(y_nnz) + 
      '\nComponente in frecventa dupa cuantizare: ' + str(y_jpeg_nnz))

# exercitiul 1
height, width = X.shape
X_jpeg = np.zeros_like(X)

def zigzag_order(N=8):
    order = []
    for s in range(2 * N - 1):  
        if s % 2 == 0:
            for i in range(min(s, N-1), max(-1, s-N), -1):
                j = s - i
                order.append((i, j))
        else:
            for i in range(max(0, s-(N-1)), min(s+1, N)):
                j = s - i
                order.append((i, j))
    return order

def zigzag_scan(block):
    N = block.shape[0]
    order = zigzag_order(N)
    return np.array([block[i, j] for (i, j) in order])

def inverse_zigzag(vec, N=8):
    block = np.zeros((N, N))
    order = zigzag_order(N)
    for idx, (i, j) in enumerate(order):
        block[i, j] = vec[idx]
    return block

def build_huffman_tree(freq):
    heap = []
    counter = 0  # tie-breaker

    for symbol, weight in freq.items():
        heappush(heap, (weight, counter, [[symbol, ""]]))
        counter += 1

    while len(heap) > 1:
        w1, _, list1 = heappop(heap)
        w2, _, list2 = heappop(heap)

        for p in list1:
            p[1] = "0" + p[1]
        for p in list2:
            p[1] = "1" + p[1]

        heappush(heap, (w1 + w2, counter, list1 + list2))
        counter += 1

    return heappop(heap)[2]

def huffman_encode(data):
    freq = Counter(data)
    huff = build_huffman_tree(freq)
    codebook = {symbol: code for symbol, code in huff}
    encoded = "".join(codebook[s] for s in data)
    return encoded, codebook

def huffman_decode(encoded, codebook):
    inv = {v: k for k, v in codebook.items()}
    decoded = []
    buffer = ""

    for bit in encoded:
        buffer += bit
        if buffer in inv:
            decoded.append(inv[buffer])
            buffer = ""
    return decoded

all_coeffs = [] 
blocks_count = 0

for i in range(0, height, 8):
    for j in range(0, width, 8):
        block = X[i:i+8, j:j+8]
        if block.shape != (8, 8):
            continue  # ignora blocurile incomplete
        Y_block = dctn(block)
        Y_quant = Q_jpeg * np.round(Y_block / Q_jpeg)
        
        zz = zigzag_scan(Y_quant)
        all_coeffs.extend(zz.tolist()) 
        blocks_count += 1

        block_jpeg = idctn(Y_quant)
        X_jpeg[i:i+8, j:j+8] = block_jpeg

plt.subplot(121).imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.subplot(122).imshow(X_jpeg, cmap=plt.cm.gray)
plt.title('JPEG Compressed')
plt.show()

encoded_stream, codebook = huffman_encode(all_coeffs)
# print(encoded_stream)  
decoded_coeffs = huffman_decode(encoded_stream, codebook)

# Reconstruim blocurile
X_decoded = np.zeros_like(X)
idx = 0
block_index = 0

for i in range(0, height, 8):
    for j in range(0, width, 8):
        if block_index >= blocks_count:
            break

        block_vec = decoded_coeffs[idx:idx+64]
        idx += 64

        block_vec = np.array(block_vec, dtype=float)
        block_freq = inverse_zigzag(block_vec)

        block_freq = block_freq * Q_jpeg
        block_spatial = idctn(block_freq)

        X_decoded[i:i+8, j:j+8] = block_spatial
        block_index += 1

plt.subplot(121).imshow(X_jpeg, cmap="gray")
plt.title("JPEG (fara Huffman)")
plt.subplot(122).imshow(X_decoded, cmap="gray")
plt.title("Decodat din Huffman")
plt.show()

# exercitiul 2
def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y  = 0.299*R + 0.587*G + 0.114*B
    Cb = -0.1687*R - 0.3313*G + 0.5*B + 128
    Cr = 0.5*R - 0.4187*G - 0.0813*B + 128
    return np.stack((Y, Cb, Cr), axis=-1)

def ycbcr_to_rgb(img):
    Y, Cb, Cr = img[..., 0], img[..., 1] - 128, img[..., 2] - 128
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    rgb = np.stack((R, G, B), axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)

def jpeg_channel(channel, Q):
    h, w = channel.shape
    compressed = np.zeros_like(channel)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = channel[i:i+8, j:j+8]
            if block.shape != (8, 8):
                continue
            Y_block = dctn(block, type=2)
            Y_quant = Q * np.round(Y_block / Q)
            block_jpeg = idctn(Y_quant, type=2)
            compressed[i:i+8, j:j+8] = block_jpeg
    return compressed


X_color = misc.face()
X_ycbcr = rgb_to_ycbcr(X_color)
Q_jpeg = np.array(Q_jpeg)  

Y_comp = jpeg_channel(X_ycbcr[..., 0], Q_jpeg)
Cb_comp = jpeg_channel(X_ycbcr[..., 1], Q_jpeg)
Cr_comp = jpeg_channel(X_ycbcr[..., 2], Q_jpeg)

X_jpeg_color = ycbcr_to_rgb(np.stack((Y_comp, Cb_comp, Cr_comp), axis=-1))

plt.subplot(121).imshow(X_color)
plt.title('Original')
plt.subplot(122).imshow(X_jpeg_color)
plt.title('JPEG Color Compressed')
plt.show()

# exercitiul 3
def compute_mse(original, compressed):
    return np.mean((original - compressed) ** 2)

def jpeg_compress_until_mse(X, Q_base, mse_target):
    Q_base = np.array(Q_base)
    Q_scale = 1 # rata de compresie
    step=0.01
    mse = float('inf')
    best_result = None

    while mse > mse_target:
        Q_scaled = Q_scale * Q_base
        X_jpeg = np.zeros_like(X)

        for i in range(0, X.shape[0], 8):
            for j in range(0, X.shape[1], 8):
                block = X[i:i+8, j:j+8]
                if block.shape != (8, 8):
                    continue
                Y_block = dctn(block, type=2)
                Y_quant = Q_scaled * np.round(Y_block / Q_scaled)
                block_jpeg = idctn(Y_quant, type=2)
                X_jpeg[i:i+8, j:j+8] = block_jpeg

        mse = compute_mse(X, X_jpeg)
        print("rata de compresie " + str(Q_scale) + " MSE " + str(mse))
        if mse <= mse_target:
            best_result = X_jpeg
            break
        Q_scale -=step  # se schimba compresia

    return best_result, Q_scale, mse

mse_target = 1.01  
X = misc.ascent()
X_jpeg_mse, Q_final, mse_final = jpeg_compress_until_mse(X, Q_jpeg, mse_target)

plt.subplot(121).imshow(X, cmap='gray')
plt.title('Original')
plt.subplot(122).imshow(X_jpeg_mse, cmap='gray')
plt.title(f'JPEG MSE≤{mse_target} (Q×{Q_final})')
plt.show()

print(f"MSE final: {mse_final:.2f}, factor de cuantizare: {Q_final}")


# exercitiul 4
import cv2

video_path = 'video.mp4'  
cap = cv2.VideoCapture('video.mp4')
frames_color = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    frames_color.append(frame)
cap.release()

compressed_color_frames = []
for frame in frames_color:
    ycbcr = rgb_to_ycbcr(frame)
    Y_comp = jpeg_channel(ycbcr[..., 0], Q_jpeg)
    Cb_comp = jpeg_channel(ycbcr[..., 1], Q_jpeg)
    Cr_comp = jpeg_channel(ycbcr[..., 2], Q_jpeg)
    compressed_rgb = ycbcr_to_rgb(np.stack((Y_comp, Cb_comp, Cr_comp), axis=-1))
    compressed_color_frames.append(compressed_rgb)

for i in range(3):
    plt.subplot(121).imshow(frames_color[i])
    plt.title('Original Frame')
    plt.subplot(122).imshow(compressed_color_frames[i])
    plt.title('Compressed Frame')
    plt.show()

out = cv2.VideoWriter('compressed_color.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (frames_color[0].shape[1], frames_color[0].shape[0]))
for f in compressed_color_frames:
    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
    out.write(bgr)
out.release()
