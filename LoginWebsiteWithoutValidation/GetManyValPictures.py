import requests

img_url = 'http://202.204.121.41:8080/reader/captcha.php'
for i in range(4000):
    print(i,end='\t')
    fr = open('./Validation_Codes_dir/%d.gif' % (i), 'wb')
    valCode = requests.get(img_url, stream=True, verify=False)
    fr.write(valCode.raw.read())
    fr.close()
