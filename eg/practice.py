# import urllib.request
# import urllib.parse

# #x = urllib.request.urlopen('https://www.google.co.in')
# #print(x.read())

# '''
# url = 'https://pythonprogramming.net/search/'
# values = {'q':'basic'}

# data = urllib.parse.urlencode(values)
# data = data.encode('utf-8')
# req = urllib.request.Request(url,data)
# resp = urllib.request.urlopen(req)
# respData = resp.read()

# print(respData)
# '''
# '''
# try:
#     x = urllib.request.urlopen('https://www.google.com/search?q=test')
#     print(x.read())
# except exception as e:
#     print(str(e))
# '''
# #this code bypasses 403 forbidden error
# try:
#     url = 'https://www.google.com/search?q=test'
#     headers = {}
#     headers['User-Agent'] = 'chrome/24.0.1312.27'
#     req = urllib.request.Request(url,headers=headers)
#     resp = urllib.request.urlopen(req)
#     respData = resp.read()

#     savefile = open('withheader.txt','w')
#     savefile.write(str(respData))
#     savefile.close()
# except exception as e:
#     print(str(e))

import sentiment_mod as s
print(s.sentiment("i dont like you fuck u u r bad"))
print(s.sentiment(" my car is so great i love it but it is not good"))
