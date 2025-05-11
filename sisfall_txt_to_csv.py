import os
import pandas as pd
import stat

filePath = "./SisFall_dataset/"

#print(os.listdir(filePath))
for i,j,k in os.walk(filePath):
    if j == []:
        new_dir = i.replace('/', '_csv/')
        os.makedirs(new_dir, exist_ok=True)
        os.chmod(new_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH| stat.S_IXOTH)
        print(i.replace('/','_csv/'))
        for filesk in k:
            if 'txt' == filesk[-3:]:
                newfilename = i.replace('/','_csv/')+'/'+filesk.replace('txt','csv')
                print(newfilename)
                csvTbl = pd.read_csv(i+'/'+filesk,header=None)
                csvTbl.iloc[:,8] = csvTbl.iloc[:,8].str.replace(';','')
                csvTbl = csvTbl.astype('float')
                csvTbl.to_csv(newfilename,encoding='GBK')