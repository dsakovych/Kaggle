import glob, os


def rename(dir, pattern, titlePattern):
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename, 
                  os.path.join(dir, titlePattern % title + ext))


if __name__=='__main__':
    rename(os.path.join(os.getcwd(), 'test_stg2'), r'*.jpg', r"test_stg2/%s")
