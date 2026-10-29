"""Read public RAR members through system libarchive; never unpack arbitrary paths."""
import contextlib
import ctypes as C
import ctypes.util
from pathlib import Path


def library():
    lib=C.CDLL(C.util.find_library('archive'))
    for name,args,result in [
        ('archive_read_new',[],C.c_void_p),
        ('archive_read_support_filter_all',[C.c_void_p],C.c_int),
        ('archive_read_support_format_all',[C.c_void_p],C.c_int),
        ('archive_read_open_filename',[C.c_void_p,C.c_char_p,C.c_size_t],C.c_int),
        ('archive_read_next_header',[C.c_void_p,C.POINTER(C.c_void_p)],C.c_int),
        ('archive_read_data',[C.c_void_p,C.c_void_p,C.c_size_t],C.c_ssize_t),
        ('archive_read_data_skip',[C.c_void_p],C.c_int),
        ('archive_read_free',[C.c_void_p],C.c_int),
        ('archive_error_string',[C.c_void_p],C.c_char_p),
        ('archive_entry_pathname',[C.c_void_p],C.c_char_p),
        ('archive_entry_size',[C.c_void_p],C.c_longlong),
        ('archive_entry_filetype',[C.c_void_p],C.c_uint)]:
        f=getattr(lib,name);f.argtypes=args;f.restype=result
    return lib


@contextlib.contextmanager
def members(path):
    lib=library();archive=lib.archive_read_new()
    try:
        lib.archive_read_support_filter_all(archive);lib.archive_read_support_format_all(archive)
        result=lib.archive_read_open_filename(archive,str(Path(path)).encode(),10240)
        if result<0:raise RuntimeError(lib.archive_error_string(archive).decode())
        def iterator():
            entry=C.c_void_p()
            while True:
                result=lib.archive_read_next_header(archive,C.byref(entry))
                if result==1:break
                if result<0:raise RuntimeError(lib.archive_error_string(archive).decode())
                name=lib.archive_entry_pathname(entry).decode('utf8')
                size=lib.archive_entry_size(entry)
                def read():
                    chunks=[];buffer=C.create_string_buffer(1024*1024)
                    while True:
                        n=lib.archive_read_data(archive,buffer,len(buffer))
                        if n==0:break
                        if n<0:raise RuntimeError(lib.archive_error_string(archive).decode())
                        chunks.append(buffer.raw[:n])
                    return b''.join(chunks)
                yield dict(member=name,bytes=size,filetype=int(lib.archive_entry_filetype(entry))),read
                lib.archive_read_data_skip(archive)
        yield iterator()
    finally:lib.archive_read_free(archive)
