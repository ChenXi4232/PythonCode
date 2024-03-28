import os


def mixTss(name):
    # string前面加上‘r’,是为了告诉编译器这个string是个raw string，不要转义 backslash '\' 。
    com = r'D:\\ffmpeg\\bin\\ffmpeg.exe -f concat -safe 0 -i D:\\ProgramData\\study\\mov\\order.m3u8 -c copy D:\\ProgramData\\study\\mov\\{}.mp4'.format(
        name)
    os.system(com)


mixTss("hello")
print("合并完成！")
