import os

backup_dir = '/home/marat/Desktop/backups'
key = '/home/marat/.ssh/amazon-g2.pem'

servers = [('ec2-54-152-147-40.compute-1.amazonaws.com', 'amazon_5x5shrinklarger.pkl'),
           ('ec2-52-1-228-222.compute-1.amazonaws.com', 'amazon_train2all.pkl'),
           ('ec2-54-165-128-89.compute-1.amazonaws.com', 'amazon_5x5shrink.pkl'),
           ('ec2-52-0-251-142.compute-1.amazonaws.com', 'amazon_5x5resizelarger.pkl')]

for i, (server, model_name) in enumerate(servers):
    dir = os.path.join(backup_dir, str(i))
    if not os.path.isdir(dir):
        os.mkdir(dir)
    cmd = 'scp -i %s ubuntu@%s:~/subway-plankton/best_%s %s/' % (key, server, model_name, dir)
    print(cmd)
    os.system(cmd)

    cmd = 'scp -r -i %s ubuntu@%s:~/subway-plankton/logs %s/' % (key, server, dir)
    print(cmd)
    os.system(cmd)