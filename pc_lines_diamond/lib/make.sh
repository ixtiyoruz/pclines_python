# gcc -shared -o mx_raster_space.so mx_raster_space.cpp
gcc -c -fPIC -Werror mx_raster_space.cpp -o mx_raster_space.o
gcc -shared -Wl,-soname,mx_raster_space.so -o mx_raster_space.so  mx_raster_space.o

gcc -c -fPIC -Werror mx_lines.cpp -o mx_lines.o
gcc -shared -Wl,-soname,mx_lines.so -o mx_lines.so  mx_lines.o