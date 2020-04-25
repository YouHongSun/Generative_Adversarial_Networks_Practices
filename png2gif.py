import os
import imageio
def gif_generate(folder,name='rig.gif'):
    images=[]
    filenames=os.listdir(folder)
    for i in range(len(filenames)):
        filename=str(i+1)+'.png'
        images.append(imageio.imread(folder+'/'+filename))
    imageio.mimsave(folder+'/'+name,images)
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str)
    parser.add_argument("--name", type=str)
    #parser.add_argument("--nice", dest="nice", action="store_true")
    #parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args
if __name__=='__main__':
    import argparse
    args=get_args()
    if not args.folder==None:
        folder=args.folder
        if not args.name==None:
            name=args.name
            gif_generate(folder,name)
        else:
            gif_generate(folder)
