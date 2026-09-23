import scipy.io as sio

chan = sio.loadmat('dataset/chan.mat')
print("Keys in chan:", chan.keys())
chan_data = chan['chan']
print("chan shape:", chan_data.shape)
print("Type of first element:", type(chan_data[0,0]))
print("Length of chan_data:", len(chan_data))

try:
    chan_names = [ch[0][0] for ch in chan_data]
    print(f"Names (first 5): {chan_names[:5]}")
    print(f"Total channels: {len(chan_names)}")
except Exception as e:
    print("Error parsing:", e)

# Or maybe it's 1x60 cell array
try:
    if chan_data.shape == (1, 60) or chan_data.shape == (60, 1):
        chan_data = chan_data.flatten()
        chan_names = [ch[0] for ch in chan_data]
        print(f"Names flattened (first 5): {chan_names[:5]}")
        print(f"Total channels flattened: {len(chan_names)}")
except Exception as e:
    print("Error parsing flattened:", e)
