from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

# consumer key, consumer secret, access token, access secret.
ckey = "DzE0QAg84K6vTbYmehLT7IPwB"
csecret = "NUFKoA9aFRLkZcKNQ1RQtuIzCO7tCVyweDa2BmDNqFDS0ySJDb"
atoken = "150562066-E6iIvJNhIaYJMi2SRAJKT3EOyafdFp8nqNrwz5d3"
asecret = "crXeSSc0eABU8SKTsmrjePu00vmTZySvjkTMij5VfANr0"

class listener(StreamListener):
    def on_data(self,data):
        all_data = json.loads(data)

        tweet = all_data["text"]
        sentiment_value,confidence = s.sentiment(tweet)
        
        print(tweet,sentiment_value,confidence)

        if confidence*100>80:
        	output = open("twitter-out.txt","a")
        	output.write(sentiment_value)
        	output.write('\n')
        	output.close()
        elif confidence*100<80:
            output = open("twitter-out.txt","a")
            if sentiment_value == "neg":
                output.write("pos")
                output.write('\n')
                output.close()
          

        return True

    def on_error(self, status):
        print(status)
        print('error')


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["warm"])
