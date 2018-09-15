import os

if __name__ == '__main__':
    # 原始文件
    with open("data/out_n.txt", "w", encoding="utf-8") as fr:
        with open("data/out.txt", encoding="utf-8") as fh:
            lines = fh.readlines()
            for line in lines:
                line = line.split(":")
                fr.write(line[0]+"\t"+line[1])