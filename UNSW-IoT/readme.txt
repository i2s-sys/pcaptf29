DATA_DIM = 72
OUTPUT_DIM = 29
TRAIN_FILE = '../train_data.csv'
TEST_FILE = '../test_data.csv'

72 feature:
到达时间差与持续时间：fwd/bwd/双向的 IAT 均值/最小/最大/标准差各 4×3，共 12 个，外加流持续时间。
TCP 窗口统计：fwd/bwd/双向的窗口总和、均值、最小、最大、标准差，各 5×3。
包数量与速率：fwd/bwd/双向包数，bwd/fwd 比例，fwd/bwd/双向每秒包速率，共 7。
包长度与速率：fwd/bwd/双向的总长度、均值、最小、最大、标准差，各 5×3；bwd/fwd 长度比；fwd/bwd/双向每秒长度速率。
TCP 标志计数：FIN、SYN、RST、PSH、ACK、URG、CWE、ECE 共 8 个；此外 fwd/bwd 的 PSH 计数、URG 计数各 2×2。
头部长度：fwd/bwd/双向头部总长度以及与总长度的比值，共 6。
最后一列为 label（设备索引，对应 MAC/IP 列表中的位置）。