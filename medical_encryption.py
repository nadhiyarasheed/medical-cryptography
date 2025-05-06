# -*- coding: utf-8 -*-
"""medical encryption.ipynb

INSTALLING MODULES
"""

!pip uninstall crypto
!pip uninstall pycryptodome
!pip install pycryptodome
!pip install cryptography

from hashlib import sha256
import base64
from Crypto import Random
from Crypto.Cipher import AES
import pandas as pd
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_private_key
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization, hashes
!pip install stepic

"""KEY GENERATION"""

def gen_key():
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048,
backend=default_backend())


public_key = private_key.public_key()
return private_key, public_key
def save_pvkey(pk,
filename): pem =
pk.private_bytes(
encoding=serialization.Encoding.PEM,
format=serialization.PrivateFormat.TraditionalOpenSSL,
encryption_algorithm=serialization.NoEncryption()
)
with open(filename, 'wb') as
pem_out: pem_out.write(pem)

def save_pukey(pk, filename):
pem =
public_key.public_bytes(
encoding=serialization.Encoding.PEM,
format=serialization.PublicFormat.SubjectPublicKeyInfo
)
with open(filename, 'wb') as
pem_out: pem_out.write(pem)

private_key, public_key = gen_key()


save_pvkey(private_key, 'private_key')
save_pukey(public_key, 'public_key')
print("private key and public key
generated.")
from hashlib import sha256
import base64
from Crypto import Random
from Crypto.Cipher import
AES import pandas as pd
from cryptography.hazmat.primitives.serialization import
load_pem_private_key from cryptography.hazmat.primitives.serialization
import load_pem_public_key from cryptography.hazmat.backends import
default_backend
from cryptography.hazmat.primitives.asymmetric
import rsa from cryptography.exceptions import
InvalidSignature
from cryptography.hazmat.primitives import serialization,
hashes from cryptography.hazmat.primitives.asymmetric
import padding from PIL import Image
import stepic

"""DNA CRYPTOGRAPHY"""

DNA_data =

"words":["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","
T","U", "V","W","X","Y","Z"," ",",",".",":","0","1","2","3","4","5","6","7","8","9"],

"DNA_code":
["CGA","CCA","GTT","TTG","GGC","GGT","TTT","CGC","ATG","AGT","AAG","TGC"
,"TCC","TCT","GGA","GTG","AAC","TCA","ACG","TTC","CTG","CCT","CCG","CTA",
"AAA","CTT","ACC","TCG","GAT","GCT","ACT","TAG","ATA","GCA","GAG","AGA",
"TTA","ACA","AGG","GCG"]
}
DNA_df = pd.DataFrame.from_dict(DNA_data)
#print(DNA_df)
message = input("Please enter your message: ")
#Name : George Mendes, Gender : Male, Birthdate : 5.9.1995, SSN : 15657834939, Medical
History
: Diabetes, Diagnosis : broken arm
DNA_crypto_message = ""
word = message.upper()



for i in word:
DNA_crypto_message+= str(DNA_df.loc[ DNA_df['words'] == i , 'DNA_code' ].iloc[0])
print(DNA_crypto_message)

"""AES CRYPTOGRAPHY"""

#block size
=16
#AES-128
BS = 16
#data should be a multiple of 16 bytes n length. Pad the buffer if it is not and include the size of
the data at the beginning of the output.
pad = lambda s: bytes(s + (BS - len(s) % BS) * chr(BS - len(s) % BS),
'utf-8') unpad = lambda s : s[0:-ord(s[-1:])]

class AESCipher:


def init ( self, key ):
#generate key
self.key = bytes(key, 'utf-8')


def encrypt( self, raw ):

raw = pad(raw)
#initialization vector - 16 bytes
iv = Random.new().read( AES.block_size )
#MODE_CBC - cipher-block chaining - each plaintext block gets XOR-ed with the previous
ciphertext prior to encryption
cipher = AES.new(self.key, AES.MODE_CBC,
iv ) return base64.b64encode( iv +
cipher.encrypt( raw ) )

cipher = AESCipher('LKHlhb899Y09olUi')
AES_encrypted_message =
cipher.encrypt(DNA_crypto_message)

print(AES_encrypted_message)

"""DIGITAL SIGNATURE"""

def load_pvkey(filename):
with open(filename, 'rb') as pem_in:
pemlines = pem_in.read()

private_key = load_pem_private_key(pemlines, None,
default_backend()) return private_key

message = AES_encrypted_message
private_key = load_pvkey("private_key")



signature = private_key.sign(message,
padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
salt_length=padding.PSS.MAX_LENGTH),hashes.SH
A256())

"""SECRET MESSAGE GENERATION"""

im = Image.open('original_image.jpg')
#Encode some text into your Image file and save it in another file
secret_msg = AES_encrypted_message + bytes("SIGNATURE", 'utf-8') + signature

"""LBS STEGANOGRAPHY"""

im1 = stepic.encode(im,
secret_msg)
im1.save('encoded_image.png',
'PNG')

"""PIXEL"""

import stepic
from PIL import Image


# Load the original image
im = Image.open("original_image.jpg")


# Define the secret message and encode it as bytes
secret_msg = "This is a secret message."
secret_msg_bytes = secret_msg.encode('utf-8')
# Encode the secret message into the
image im1 = stepic.encode(im,
secret_msg_bytes)

# Pixelate the image by resizing with a low resolution
pixelated_im = im1.resize((im1.size[0] // 10, im1.size[1] // 10), Image.NEAREST)


# Save the pixelated encoded image
pixelated_im.save('pixelated_encoded_image.png', 'PNG')

print("Secret message encoded and pixelated successfully.")



import cv2
import numpy as np
from math import
log10,sqrt def
PSNR(original, steg):
mse = np.mean((original - steg)
** 2) if(mse == 0):
return 100
max_pixel = 255.0
psnr = 20 * log10(max_pixel /
sqrt(mse)) return psnr

original=cv2.imread("original_image.jpg
")
steg=cv2.imread("encoded_image.png")
value = PSNR(original,steg)

print(f"PSNR value: {value} dB")

"""IMAGE COMPRESSION"""

import sys, os, time, numpy,
pywt import matplotlib.pyplot
as plt from PIL import Image

def wavelet_transform(data, threshold):
wavelet_type = 'haar'
clean_coef = list()
compose = list()

cA2, cD2, cD1 = pywt.wavedec2(data, wavelet_type, level=2)
clean_coef.append(cA2)
clean_coef.append(cD2)
for c in cD1:
compose.append(numpy.where(((c<(-threshold)) | (c>threshold)),
c, 0)) clean_coef.append(tuple(compose))

t = pywt.waverec2(clean_coef,
wavelet_type) values = t.astype(int)
return values

def create_image(image, values,
threshold): matrix = list()
for value in
values: row =
list()
for v in value:
row.append((int(v), int(v), int(v)))
matrix.append(row)
width, height = image.size
new_image = Image.new('RGB', (width,
height)) new = new_image.load()
for w in
range(width): for
h in
range(height):
new[w, h] = matrix[h][w]


image_name = str(threshold) +
'.png'
new_image.save(image_name)
return new_image

def grayscale(image):
width, height = image.size
pixels = image.load()


for w in
range(width): for
h in
range(height):
r, g, b = pixels[w,
h] gray =
(r+g+b)//3
pixels[w, h] = (gray, gray,
gray) return image

def
get_rows_values(image
): width, height =
image.size pixels =
image.load() matrix =
list()


for j in range(height):
  row = list()
for i in range(width):
pixel_value = pixels[i,
j][0]
row.append(pixel_value)
matrix.append(row)

array =
numpy.array(matrix)
return array

def compress(image_path, threshold):
image = Image.open(image_path).convert('RGB')
image = grayscale(image)

data = get_rows_values(image)
values = wavelet_transform(data, threshold)


newimage = create_image(image, values,
threshold) return
compressed_percentage(image_path, threshold)
def compressed_percentage(image_path, threshold):
original_size = os.path.getsize(image_path)
image_name = str(threshold) + '.png'
final_size = os.path.getsize(image_name)
percentage = 100 -
(final_size*100)//float(original_size) print ('Image
compressed at %0.2f%%' % percentage) return
percentage


def main():
image_path = "encoded_image.png"
time_list = list()
percentages_list = list()
thresholds_list = list()
for threshold in range(0, 200, 20):
start_time = time.time()
compressed_percentage = compress(image_path, threshold)
end_time = time.time()
process_time = end_time - start_time
time_list.append(process_time)
percentages_list.append(compressed_percentage)
thresholds_list.append(threshold)
p = plt.plot(thresholds_list, percentages_list, 'bo-', label='Percentage')
plt.legend(loc='upper left', numpoints=1)
plt.ylabel('Percentage')
plt.xlabel('Threshold value')
plt.title('Percentage vs. Threshold value')
plt.show()
average_time = sum(time_list)//len(time_list)
print ('The average time is', average_time)
if name == ' main
': main()
import cv2
import numpy as np
from math import
log10,sqrt def
PSNR(original, steg):
mse = np.mean((original - steg)
** 2) if(mse == 0):
return 100
max_pixel = 255.0
psnr = 20 * log10(max_pixel /
sqrt(mse)) return psnr
original=cv2.imread("original_image.jpg
") steg=cv2.imread("180.png")
value = PSNR(original,steg)
print(f"PSNR value: {value} dB")

"""TARGET END"""

from hashlib import sha256
import base64
from Crypto import Random
from Crypto.Cipher import
AES import pandas as pd
from cryptography.hazmat.primitives.serialization import
load_pem_private_key from cryptography.hazmat.primitives.serialization
import load_pem_public_key from cryptography.hazmat.backends import
default_backend
from cryptography.hazmat.primitives.asymmetric
import rsa from cryptography.exceptions import
InvalidSignature
from cryptography.hazmat.primitives import serialization,
hashes from cryptography.hazmat.primitives.asymmetric
import padding from PIL import Image
import stepic

"""DECRYPTION"""

im = Image.open('encoded_image.png')
stegoImage = stepic.decode(im)

ind_sep = stegoImage.find('SIGNATURE')
message = bytes(stegoImage[:ind_sep],'utf-8')
signature = bytes(stegoImage[ind_sep+9:],'latin1')


def load_pukey(filename):
with open(filename, 'rb') as pem_in:
pemlines = pem_in.read()
public_key = load_pem_public_key(pemlines,
default_backend()) return public_key

public_key =
load_pukey("public_key") try:
public_key.verify(signature, message, padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
salt_length=padding.PSS.MAX_LENGTH),hashes.SHA256())
print(message)
except
InvalidSignature:
print('Invalid!')


BS = 16
pad = lambda s: bytes(s + (BS - len(s) % BS) * chr(BS - len(s) % BS),
'utf-8') unpad = lambda s : s[0:-ord(s[-1:])]

class AESCipher:


def init ( self, key ):
self.key = bytes(key,
'utf-8')

def decrypt( self, enc ):
enc =
base64.b64decode(enc)
iv = enc[:16]
cipher = AES.new(self.key, AES.MODE_CBC, iv )
return unpad(cipher.decrypt( enc[16:] )).decode('utf8')
cipher =AESCipher('LKHlhb899Y09olUi')
AES_decrypted =
cipher.decrypt(message)
print(AES_decrypted)

DNA_data =
{
"words":["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","
U", "V","W", "X","Y","Z"," ",",",".",":","0","1","2","3","4","5","6","7","8","9"],

"DNA_code":["CGA","CCA","GTT","TTG","GGC","GGT","TTT","CGC","ATG","AGT
","AAG","TGC","TCC","TCT","GGA","GTG","AAC","TCA","ACG","TTC","CTG","CCT","C
CG","CTA","AAA","CTT","ACC","TCG","GAT","GCT","ACT","TAG","ATA","GCA","GAG",
"AGA","TTA","ACA","AGG","GCG"]
}


DNA_df = pd.DataFrame.from_dict(DNA_data)l = [AES_decrypted[i:i+3]
for i in range(0, len(AES_decrypted), 3)]
original_message = ""
for i in l:
original_message+= str(DNA_df.loc[ DNA_df['DNA_code'] == i , 'words' ].iloc[0])
print("The secret message is: ",original_message.lower())
from Crypto.Cipher import AES
from Crypto.Util.Padding import
unpad from Crypto import Random
import base64

class AESCipher:
def _init_(self, key):
self.key = key

def encrypt(self, data):
padded_data = pad(data,
AES.block_size) iv =
Random.new().read(AES.block_size)
cipher = AES.new(self.key,AES.MODE_CBC, iv) encrypted_data = iv +
cipher.encrypt(padded_data)
return base64.b64encode(encrypted_data)

def decrypt(self, encrypted_data):
encrypted_data =
base64.b64decode(encrypted_data) iv =
encrypted_data[:AES.block_size]
cipher = AES.new(self.key, AES.MODE_CBC, iv)
decrypted_data = cipher.decrypt(encrypted_data[AES.block_size:])
return unpad(decrypted_data, AES.block_size)
def decrypt_image(image_path, key):
with open(encoded_image.png, 'rb') as
f: encrypted_data = f.read()


aes_cipher = AESCipher(b'LKHlhb899Y09olUi')
decrypted_data =
aes_cipher.decrypt(encrypted_data)

return decrypted_data
