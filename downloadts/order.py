import m3u8
data = m3u8.load("index0.m3u8").data
print(data)

order_ts = []
# 将所有的带https的url存入order_ts
for i in data["segments"]:
    order_ts.append("play.modujx.com" + i["uri"])

# 返回一个dict，将文件名作为key，将url作为value


def read_name_url():
    name_url = {}
    for url in order_ts:
        name = url.split("/")[-1]
        name_url[name] = url
    return name_url


list_name = read_name_url().keys()
file = open("meimo0.m3u8", 'w')
for name in list_name:
    # file.write("file 'D:\\ProgramData\\study\\mov\\tsfiles\\" + name + "'")
    file.write(fr"file 'D:\study\mov\meimo\{name}'")
    file.write("\n")
file.close()

print(read_name_url())
